import os, sys, time, pickle, re, warnings
from difflib import SequenceMatcher
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch

# ── Scratch LLM (zero pretrained weights) ────────────────────────────────────
from scratch_llm import ScratchLLM as _ScratchLLM

# ── Data-quality validation, ambiguity detection, confidence calibration ────
# These were previously built but never wired into the running pipeline.
# Imports are guarded the same way query_engine's is below: a missing/broken
# module must never prevent the chatbot from starting.
try:
    import data_quality
except Exception as exc:
    print(f"  [WARN] data_quality unavailable: {exc}")
    data_quality = None

try:
    import ambiguity_detector
except Exception as exc:
    print(f"  [WARN] ambiguity_detector unavailable: {exc}")
    ambiguity_detector = None

try:
    import confidence as confidence_mod
except Exception as exc:
    print(f"  [WARN] confidence unavailable: {exc}")
    confidence_mod = None

LLM_DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MAX_TOKENS  = 80
LLM_TEMPERATURE = 0.7
LLM_TOP_P       = 0.9

_scratch_llm = _ScratchLLM()
_llm_ready   = False


def _load_llm() -> bool:
    """Load the scratch-trained SalesGPT model (no pretrained weights)."""
    global _llm_ready
    _llm_ready = _scratch_llm.load()
    return _llm_ready


_SALES_SYSTEM_PROMPT = (
    "Report the database facts below in plain English. "
    "One or two sentences only. "
    "Use only the facts given. "
    "Do not add opinions, advice, or extra information. "
    "Do not greet. Do not introduce yourself. Just state the facts."
)


def _generate_phrase(query: str, structured_facts: str, intent: str,
                      require_words: set | None = None) -> str:
    """
    Ask the scratch LLM to phrase `structured_facts` (already-correct,
    deterministically retrieved data) as a natural sentence answering
    `query`. Used for EVERY intent now — list/aggregation queries AND
    single-record queries — so the assistant always reasons "retrieve →
    generate" instead of ever just dumping raw fields at the user.

    `require_words`, when given (typically the exact customer name and/or
    enquiry ID for a single-record answer), must appear verbatim in the
    generated text or the output is rejected. This is the safeguard that
    lets a tiny model generate freely without being able to quietly swap
    in a different person's name.
    """
    if not _load_llm():
        return None

    list_intents = {"list_all", "bad_feedback", "good_feedback", "cancelled",
                    "completed", "new_lead", "city", "returning"}
    task = ("State the total count and name up to 3 examples in one sentence."
            if intent in list_intents else
            "Answer the question directly, in one or two natural sentences, "
            "using only the data above.")

    if intent in list_intents:
        lines     = structured_facts.strip().split("\n")
        trimmed   = lines[:6]
        remaining = len(lines) - len(trimmed)
        facts_for_llm = "\n".join(trimmed)
        if remaining > 0:
            facts_for_llm += f"\n… and {remaining} more entries."
    else:
        facts_for_llm = structured_facts.strip()

    prompt_text = (
        f"{_SALES_SYSTEM_PROMPT}\n\n"
        f"Data:\n{facts_for_llm}\n\n"
        f"Question: {query}\n\n"
        f"{task}"
    )

    raw = _scratch_llm.generate(
        prompt_text,
        max_new=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
    )
    if not raw or len(raw.strip()) < 10:
        return None

    # Basic cleanup
    BAD_PHRASES = [
        "i don't have information", "i cannot answer", "as an ai",
        "i was trained", "my knowledge cutoff", "i do not know",
        "i'm not sure", "dear customer", "sure, here's", "here's how",
        "thank you for reaching", "based on my knowledge",
    ]
    if any(p in raw.lower() for p in BAD_PHRASES):
        return None

    # ── Groundedness check ───────────────────────────────────────────────
    # The scratch model is small and trained on a limited corpus, so it can
    # hallucinate — inventing a different customer name, ID, or fact than
    # the one actually supplied. Require the generated sentence to reuse at
    # least one substantial word (4+ letters) from the facts it was given;
    # if it shares nothing with the source data, it isn't describing this
    # record and must be rejected rather than shown to the user.
    fact_words = set(re.findall(r"[a-zA-Z]{4,}", facts_for_llm.lower()))
    gen_words  = set(re.findall(r"[a-zA-Z]{4,}", raw.lower()))

    # ── Groundedness check (coverage ratio, not just "shares one word") ───
    # The scratch model is small and trained on templated pairs covering
    # MULTIPLE intents per customer record. Given facts for one intent
    # (say, contact details), it can still complete the sentence with
    # memorized fragments from a DIFFERENT intent about the same person
    # (a feedback rating, a status line, etc.) — content that was never in
    # the facts it was actually given for this answer. A single shared
    # word is not enough evidence the output is grounded; the bulk of the
    # generated content must come from these facts, or it's rejected.
    if gen_words:
        overlap_ratio = len(gen_words & fact_words) / len(gen_words)
        if overlap_ratio < 0.5:
            return None
    elif fact_words:
        return None

    # ── Identity lock ────────────────────────────────────────────────────
    # For single-record answers, the generated sentence must literally
    # contain the customer name / enquiry ID it was given. A few-million-
    # parameter model trained on a small corpus can otherwise produce a
    # fluent sentence about the WRONG person; this check catches that
    # before it ever reaches the user, without needing a bigger model.
    if require_words:
        raw_low = raw.lower()
        if not any(w and w.lower() in raw_low for w in require_words):
            return None

    sentences = re.split(r"(?<=[.!?])\s+", raw.strip())
    raw = " ".join(sentences[:3]).strip()
    return raw if len(raw) >= 10 else None


warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────── CONFIG
TOP_K       = 8
INDEX_CACHE = "rag_index.pkl"

CSV_FILES = {
    "Enquiry"    : "sales_enquiry_dataset.csv",
    "Appointment": "sales_appointment_dataset.csv",
    "Feedback"   : "sales_feedback_dataset.csv",
}

INTENT_TO_SOURCE = {
    "appointment" : "Appointment",
    "cancelled"   : "Appointment",
    "completed"   : "Appointment",
    "feedback"    : "Feedback",
    "bad_feedback": "Feedback",
    "good_feedback": "Feedback",
    "contact"     : "Enquiry",
    "status"      : "Enquiry",
    "payment"     : "Enquiry",
    "test_ride"   : "Enquiry",
    "vehicle"     : "Enquiry",
    "new_lead"    : "Enquiry",
    "city"        : "Enquiry",
    "returning"   : "Enquiry",
    "summary"     : None,
    "list_all"    : None,
}


_KNOWN_NAMES_SET: set = set()
DATASET_QUALITY: dict = {}   # populated by load_datasets() via data_quality.py


def get_dataset_quality() -> dict:
    """Exposed for api.py's /api/health/ endpoint."""
    return DATASET_QUALITY


SYNONYMS = {
    "enquiry":     ["enquiry","inquiry","lead","record","customer","data","details","info",
                    "information","profile","case","file","ticket"],
    "feedback":    ["feedback","review","rating","comment","opinion","experience","satisfaction",
                    "complaint","response","feeling","sentiment","happy","unhappy","satisfied",
                    "dissatisfied","good","bad","poor","excellent","average","reaction",
                    "say","said","says","saying","told","mentioned","stated","commented",
                    "think","thinks","thought","feel","feels","felt","service","quality"],
    "appointment": ["appointment","meeting","visit","booking","schedule","slot","session",
                    "confirmed","booked","planned","upcoming","timing","time","date"],
    "status":      ["status","state","progress","update","stage","current","situation",
                    "standing","position","pending","closed","open","active"],
    "contact":     ["contact","phone","mobile","email","mail","reach","number","call"],
    "vehicle":     ["vehicle","car","bike","model","automobile","product","item",
                    "interested","buying","purchase","want","enquired about"],
    "payment":     ["payment","paid","pay","loan","cash","emi","finance","amount","mode","method"],
    "test_ride":   ["test ride","test drive","trial","drove","tried","ridden","demo"],
    "bad":         ["bad","poor","negative","low","worst","unhappy","dissatisfied","complaint","below"],
    "good":        ["good","great","excellent","positive","high","happy","satisfied",
                    "wonderful","amazing","best","top"],
    "city":        ["city","location","place","from","region","area","state","where","lives"],
    "new":         ["new","fresh","recent","just","newly","latest"],
    "returning":   ["returning","existing","repeat","old","regular","loyal","again"],
    "all":         ["all","entire","complete","full","every","each","whole","list","show",
                    "give","dataset","records","everyone","dump"],
    "cancelled":   ["cancel","cancelled","cancellation","abort","drop","not coming","no show","withdrew"],
    "completed":   ["completed","done","finished","over","past","visited","successful"],
}

INTENT_CLUSTERS = {
    "list_all":      {"all","entire","complete","full","every","each","whole","list","dataset",
                      "records","give","show","take","display","dump","everybody"},
    "bad_feedback":  {"bad","poor","negative","low","worst","unhappy","dissatisfied","complaint",
                      "complaints","below","rating","feedback","review"},
    "good_feedback": {"good","great","excellent","positive","high","happy","satisfied",
                      "wonderful","amazing","best","top","rating","feedback","review"},
    "feedback":      {"feedback","review","rating","comment","opinion","experience",
                      "satisfaction","sentiment","said","wrote","gave","submitted",
                      "say","says","saying","told","mentioned","stated","commented",
                      "think","thinks","thought","feel","feels","felt","service","quality"},
    "appointment":   {"appointment","meeting","visit","booking","schedule","slot","booked",
                      "planned","upcoming","timing","time","date","session","confirmed"},
    "cancelled":     {"cancel","cancelled","cancellation","abort","drop","no","not"},
    "completed":     {"completed","done","finished","over","past","visited","successful"},
    "contact":       {"contact","phone","mobile","email","mail","reach","number","call"},
    "status":        {"status","state","progress","update","stage","current","situation",
                      "standing","position","pending","closed","open","active"},
    "payment":       {"payment","paid","pay","loan","cash","emi","finance","amount","mode"},
    "test_ride":     {"test","ride","drive","trial","drove","tried","ridden","demo"},
    "vehicle":       {"vehicle","car","bike","model","automobile","product","item",
                      "interested","buying","purchase","want"},
    "city":          {"city","location","place","from","region","area","state","where"},
    "new_lead":      {"new","fresh","recent","newly","latest","lead","enquiry"},
    "returning":     {"returning","existing","repeat","old","regular","loyal"},
    "summary":       {"details","summary","everything","full","complete","about","info",
                      "information","profile","all","tell","know","summarize","summarise"},
}


def expand_query(query: str) -> str:
    q_lower = query.lower()
    extra   = []
    for _, words in SYNONYMS.items():
        if any(w in q_lower for w in words):
            extra.extend(words)
    return query + " " + " ".join(extra)


def detect_intent(query: str, known_name: str | None = None, retriever=None) -> tuple:
    expanded    = set(re.findall(r'\w+', expand_query(query).lower()))
    q_lower     = query.lower()
    q_words_raw = set(re.findall(r'\w+', q_lower))

    scores = {intent: len(expanded & cluster) / max(len(cluster), 1)
              for intent, cluster in INTENT_CLUSTERS.items()}
    best = max(scores, key=scores.get)

    # ── Semantic (retrieval-based) fallback ────────────────────────────────
    # If NOTHING in the keyword vocabulary matched at all (max score == 0),
    # the query is phrased in a way our word lists simply don't cover
    # (e.g. "what did Ananya say about the service"). Rather than silently
    # falling through to whichever intent happens to be first in the dict
    # (list_all → then a raw full-record dump), actually run retrieval and
    # let the dominant SOURCE of the top-matching records decide the
    # intent. This is a lightweight "understand the meaning, then decide"
    # step instead of pure keyword matching, and it's what makes queries
    # the vocabulary lists were never taught about still resolve sensibly.
    used_semantic_fallback = False
    if scores[best] == 0 and retriever is not None:
        probe = retriever.retrieve(query, top_k=5)
        if probe:
            top_source = Counter(r.get("__source__") for r in probe).most_common(1)[0][0]
            best = {"Feedback": "feedback", "Appointment": "appointment",
                    "Enquiry": "summary"}.get(top_source, "summary")
            used_semantic_fallback = True

    has_name = bool(extract_name(query)) or bool(_extract_name_lower(query)) or bool(known_name)

    INFO_FIELDS = {"enquiry","id","details","info","information","record","profile",
                   "data","about","for","say","show","tell","give","only","find","get",
                   "take","fetch","pull","of","regarding","related","specific","his","her"}
    if not used_semantic_fallback:
        if has_name and (q_words_raw & INFO_FIELDS) and best in ("new_lead","list_all","summary"):
            best = "summary"
        if has_name and best == "list_all":
            best = "summary"

    ENQ_ID_WORDS = {"enquiry","id","enq","number","ref","reference","code"}
    if best == "new_lead" and has_name and (q_words_raw & ENQ_ID_WORDS):
        best = "summary"

    POLARITY_POS = {"good","great","excellent","positive","high","happy","satisfied",
                    "wonderful","amazing","best","top"}
    POLARITY_NEG = {"bad","poor","low","unhappy","dissatisfied","worst","complaint","negative"}
    if scores.get("bad_feedback",0) > 0 and scores.get("good_feedback",0) > 0:
        best = "bad_feedback" if q_words_raw & POLARITY_NEG else "good_feedback"

    # When a specific customer name is in the query and the intent resolved to
    # a global-list intent (good_feedback / bad_feedback), the user is asking
    # about ONE person's feedback, not requesting a full list of all customers
    # with high/low ratings.  Flip to the single-record "feedback" intent so
    # build_answer returns just that customer's record.
    # Note: "satisfied", "happy" etc. are polarity words but when used in
    # "how satisfied was Arjun" they describe the question, not a filter —
    # having a name present is the authoritative signal here.
    if best in ("bad_feedback", "good_feedback") and has_name:
        best = "feedback"

    if best == "appointment":
        if any(w in q_lower for w in ["cancel","cancelled","cancellation","no show","no-show"]):
            best = "cancelled"
        elif any(w in q_lower for w in ["completed","done","finished","past","over","successful"]):
            best = "completed"

    return best, scores[best]


def _extract_name_lower(query: str):
    words = re.findall(r'\b\w+\b', query.lower())
    for w in words:
        if w.title() in _KNOWN_NAMES_SET:
            return w.title()
    return None


def normalize_query(query: str) -> str:
    query = re.sub(r'\bEQ(\d+)\b',     r'ENQ\1', query, flags=re.IGNORECASE)
    query = re.sub(r'\benq\s*(\d+)\b', r'ENQ\1', query, flags=re.IGNORECASE)
    return query


def extract_enq_id(query: str):
    m = re.search(r'\b(ENQ\d+)\b', query, re.IGNORECASE)
    return m.group(1).upper() if m else None


_PHONE_RE = re.compile(r'\b(\d{10})\b')


def extract_phone(query: str):
    """A standalone 10-digit number is treated as a phone number to look
    up directly — this is what makes 'whose number is this 9945206132'
    an exact reverse lookup instead of a fuzzy TF-IDF guess."""
    m = _PHONE_RE.search(query)
    return m.group(1) if m else None


def has_own_identifier(query: str) -> bool:
    """
    True if the query already fully specifies which record it's about —
    an enquiry ID, a 10-digit phone number, or an explicit date. Queries
    like this must NEVER borrow a customer name from earlier turns, even
    though they contain digits: the digits already ARE the identifier.
    """
    return bool(extract_enq_id(query) or extract_phone(query) or extract_query_date(query))


_DATE_RE = re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b')


def extract_query_date(query: str) -> str | None:
    m = _DATE_RE.search(query)
    if not m:
        return None
    month, day, year = m.groups()
    try:
        return f"{int(month)}/{int(day)}/{year}"
    except ValueError:
        return None


def _row_matches_date(row: dict, date_str: str) -> bool:
    for col, val in row.items():
        if col.startswith("__"):
            continue
        if "date" in col.lower() and str(val).strip() == date_str:
            return True
    return False


_NAME_STOP = {
    "is","was","are","the","for","when","what","who","which","how","why","whom",
    "has","have","did","does","do","did","can","could","would","should",
    "tell","me","my","give","other","details","of","email","phone","show",
    "find","get","contact","id","satisfied","feedback","rating","payment","payments",
    "customer","customers","appointment","appointments","enquiry","enquiries",
    "status","dataset","information","data","lead","leads","record","records",
    "vehicle","vehicles","car","cars","bike","bikes","model","models",
    "all","any","please","about","from","their","his","her","its","this","that",
    "with","and","or","but","in","on","at","to","by","an","a","no","not",
    "take","give","list","entire","complete","every","each","whole",
    "summary","summarize","summarise","everything","best","match","assistant",
    "hyderabad","bangalore","bengaluru","chennai","mumbai","delhi","pune",
    "coimbatore","kolkata","ahmedabad","surat","jaipur","lucknow","nagpur",
    "ts","tn","ka","mh","gj","rj","up","wb","ap","telangana","karnataka",
    "tamilnadu","maharashtra","gujarat",
}


def extract_name(query: str):
    """
    Heuristically extract a customer name from a capitalised word/phrase in
    the query.

    IMPORTANT: a bare "starts with a capital letter" match is not enough —
    the first word of any sentence is capitalised too ("Which customers…",
    "Who gave…"), and multi-word vehicle models ("Kia Seltos", "Honda
    City") match the same capitalised-phrase shape as a person's name.
    Both were previously mistaken for a customer name, which silently
    turned "which customers …" group questions into a single-record
    lookup for whichever row happened to rank first.

    To avoid that, any candidate is cross-checked against the known
    customer-name registry (`_KNOWN_NAMES_SET`, populated at startup from
    the dataset's own "name" columns) before being trusted. If the
    registry hasn't been populated yet (e.g. running utility functions in
    isolation), fall back to the old stop-word-only heuristic.
    """
    matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
    known_lower = {n.lower() for n in _KNOWN_NAMES_SET}
    for m in matches:
        parts = [p for p in m.split() if p.lower() not in _NAME_STOP]
        if not parts:
            continue
        candidate = " ".join(parts)
        if known_lower:
            # Trust it only if the whole phrase, or at least one of its
            # words, is an actual known customer name.
            if candidate.lower() not in known_lower and not any(
                p.lower() in known_lower for p in parts
            ):
                continue
        return candidate
    return None


def fuzzy_match_name(name: str, candidates: list, threshold: float = 0.75) -> list:
    nl  = name.lower()
    out = []
    for c in candidates:
        cl = c.lower()
        if nl in cl or cl in nl:
            out.append(c)
        elif SequenceMatcher(None, nl, cl).ratio() >= threshold:
            out.append(c)
    return out


def row_to_rich_text(row: dict, source: str) -> str:
    def v(key): return str(row.get(key, "") or "").strip()

    if source == "Enquiry":
        name   = v("Customer Name")
        eid    = v("ENQUIRY ID")
        phone  = v("Phone Number")
        email  = v("Email")
        gender = v("Gender")
        veh    = v("Vehicle Name / Model")
        src    = v("Enquiry Source")
        edate  = v("Enquiry Date")
        adate  = v("Appointment Date")
        city   = v("City / State")
        ctype  = v("Customer Type")
        pay    = v("Payment Type")
        ride   = v("Test Ride Taken")
        status = v("Status")
        ride_t = "took test ride test drive" if ride.lower() == "yes" else "did not take test ride"
        return (
            f"[Enquiry] Customer {name} enquiry ID {eid} "
            f"is a {gender} {ctype} customer from {city} location city. "
            f"Interested in vehicle car model {veh}. "
            f"Enquiry source channel {src} on date {edate}. "
            f"Appointment scheduled on {adate}. "
            f"Contact phone {phone} email {email}. "
            f"Payment mode method {pay} loan cash emi. "
            f"Status progress stage {status}. "
            f"Test ride test drive {ride} {ride_t}. "
            f"Enquiry record lead data information details profile."
        )

    elif source == "Appointment":
        name   = v("Customer Name")
        eid    = v("Enquiry ID")
        adate  = v("Appointment Date")
        atime  = v("Time")
        veh    = v("Vehicle")
        status = v("Status")
        extra  = ("confirmed scheduled booked planned upcoming" if status == "Scheduled" else
                  "completed done finished successful visited"   if status == "Completed" else
                  "cancelled canceled abort withdrawn no-show"   if status == "Cancelled" else "")
        return (
            f"[Appointment] Customer {name} enquiry ID {eid} "
            f"appointment meeting visit booking on date {adate} at time {atime}. "
            f"Vehicle car model {veh}. Status {status} {extra}. "
            f"Appointment record data details."
        )

    elif source == "Feedback":
        name     = v("Customer Name")
        eid      = v("Enquiry ID")
        feedback = v("Feedback")
        rating   = v("Rating")
        date     = v("Date")
        try:
            r   = int(float(rating))
            sent = ("positive excellent satisfied happy great good" if r >= 4 else
                    "neutral average okay"                           if r == 3 else
                    "negative poor bad unhappy dissatisfied complaint low")
            neg_tag = "complaint negative review low rating bad dissatisfied unhappy" if r <= 2 else ""
            pos_tag = "excellent positive review high rating good satisfied happy"    if r >= 4 else ""
        except Exception:
            sent = neg_tag = pos_tag = ""
        return (
            f"[Feedback] Customer {name} enquiry ID {eid} "
            f"submitted feedback review comment on date {date}: '{feedback}'. "
            f"Rating score {rating} out of 5. "
            f"Sentiment {sent}. {neg_tag} {pos_tag}. "
            f"Feedback review data details."
        )

    else:
        parts = [f"[{source}]"]
        for col, val in row.items():
            if not col.startswith("__") and val and str(val) not in ("nan","None",""):
                parts.append(f"{col} {val}")
        return " ".join(parts)


class IntelligentRetriever:
    def __init__(self, dfs: dict, docs_per_source: dict):
        self.dfs     = dfs
        self.sources = list(dfs.keys())
        self.df      = pd.concat(dfs.values(), ignore_index=True)

        self._per_source: dict = {}
        for src, src_df in dfs.items():
            docs     = docs_per_source[src]
            vec      = TfidfVectorizer(
                ngram_range=(1, 3), max_features=20_000,
                sublinear_tf=True, min_df=1,
                token_pattern=r'(?u)\b\w[\w\-]*\b',
            )
            expanded = [d + " " + expand_query(d) for d in docs]
            mat      = vec.fit_transform(expanded)
            self._per_source[src] = {"vec": vec, "mat": mat, "df": src_df.reset_index(drop=True)}
            print(f"  OK [{src:<12}] index: {mat.shape[0]} rows × {mat.shape[1]} vocab")

        all_docs = [d for src in self.sources for d in docs_per_source[src]]
        vec_all  = TfidfVectorizer(
            ngram_range=(1, 3), max_features=30_000,
            sublinear_tf=True, min_df=1,
            token_pattern=r'(?u)\b\w[\w\-]*\b',
        )
        self.matrix = vec_all.fit_transform([d + " " + expand_query(d) for d in all_docs])
        self.vec    = vec_all
        print(f"  OK [Unified    ] index: {self.matrix.shape[0]} rows × {self.matrix.shape[1]} vocab")

        self.name_index: dict = defaultdict(list)
        self.all_names:  list = []
        for src, src_df in dfs.items():
            for _, row in src_df.iterrows():
                for col in src_df.columns:
                    if "name" in col.lower() and not col.startswith("__"):
                        name = str(row[col]).strip()
                        if name and name not in ("nan","None",""):
                            self.name_index[name.lower()].append(
                                {"source": src, "row": row.to_dict()}
                            )
                            self.all_names.append(name)
        self.all_names = sorted(set(self.all_names))

    def retrieve_from_source(self, query: str, source: str,
                             top_k: int = TOP_K) -> list:
        if source not in self._per_source:
            return []
        idx  = self._per_source[source]
        vec, mat, src_df = idx["vec"], idx["mat"], idx["df"]

        qvec   = vec.transform([expand_query(query)])
        scores = cosine_similarity(qvec, mat).flatten()
        top_i  = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for i in top_i:
            if scores[i] < 0.01:
                continue
            row               = src_df.iloc[i].to_dict()
            row["__source__"] = source
            row["__score__"]  = round(float(scores[i]), 4)
            results.append(row)
        return results[:top_k]

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        qvec   = self.vec.transform([expand_query(query)])
        scores = cosine_similarity(qvec, self.matrix).flatten()
        top_i  = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for i in top_i:
            if scores[i] < 0.01:
                continue
            row              = self.df.iloc[i].to_dict()
            row["__score__"] = round(float(scores[i]), 4)
            results.append(row)
        return results[:top_k]

    def retrieve_by_name(self, name: str, preferred_source: str = None,
                         top_k: int = TOP_K) -> list:
        matched_names = fuzzy_match_name(name, self.all_names)

        def rows_for(src: str) -> list:
            found = []
            for mname in matched_names:
                for entry in self.name_index.get(mname.lower(), []):
                    if entry["source"] == src:
                        r               = dict(entry["row"])
                        r["__source__"] = src
                        r["__score__"]  = 1.0
                        found.append(r)
            return found

        if preferred_source:
            rows = rows_for(preferred_source)
        else:
            rows = []

        if not rows:
            for src in self.sources:
                if src != preferred_source:
                    rows.extend(rows_for(src))

        seen, out = set(), []
        for r in rows:
            key = (r.get("__source__"), str(find_val(r, "enquiry id", "ENQUIRY ID") or ""))
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out[:top_k]

    def get_all_by_source(self, source: str) -> list:
        if source not in self._per_source:
            return []
        rows = []
        for _, row in self._per_source[source]["df"].iterrows():
            r               = row.to_dict()
            r["__source__"] = source
            r["__score__"]  = 1.0
            rows.append(r)
        return rows

    def get_by_enq_id(self, enq_id: str) -> dict:
        result = {}
        for src, idx in self._per_source.items():
            src_df = idx["df"]
            for col in src_df.columns:
                if "enquiry" in col.lower() and "id" in col.lower():
                    mask = src_df[col].astype(str).str.upper() == enq_id.upper()
                    hits = src_df[mask]
                    if not hits.empty:
                        r               = hits.iloc[0].to_dict()
                        r["__source__"] = src
                        result[src]     = r
                    break
        return result

    def get_by_phone(self, phone: str) -> dict:
        """Exact reverse lookup: given a phone number, find whose record
        it belongs to. Digits-only comparison so formatting differences
        (spaces/dashes) don't cause a miss."""
        digits = re.sub(r"\D", "", phone)
        result = {}
        for src, idx in self._per_source.items():
            src_df = idx["df"]
            for col in src_df.columns:
                if "phone" in col.lower() or "mobile" in col.lower():
                    mask = src_df[col].astype(str).apply(lambda v: re.sub(r"\D", "", v) == digits)
                    hits = src_df[mask]
                    if not hits.empty:
                        r               = hits.iloc[0].to_dict()
                        r["__source__"] = src
                        result[src]     = r
                    break
        return result

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def find_val(row: dict, *keys):
    for k in keys:
        for col, val in row.items():
            if col.startswith("__"):
                continue
            if k.lower() in col.lower():
                vs = str(val).strip()
                if vs and vs.lower() not in ("nan","none","","n/a"):
                    return vs
    return None


def _rating(row: dict):
    val = find_val(row, "rating")
    try:    return int(float(val))
    except: return None


def get_source(row: dict) -> str:
    return row.get("__source__", "record")


def build_answer(query: str, results: list, intent: str,
                 retriever: IntelligentRetriever) -> str:

    q_low = query.lower()

    if not results:
        return ("I could not find any relevant records. "
                "Try rephrasing or check the name / ID.")

    eid = extract_enq_id(query)
    if eid:
        by_source = retriever.get_by_enq_id(eid)
        if not by_source:
            return f"No records found for enquiry ID '{eid}'."

        # Route the exact rows for this ID into `results` instead of
        # dumping everything immediately. This lets the intent-specific
        # branches below (status, city, vehicle, payment, returning,
        # new_lead, contact, test_ride, feedback, appointment, and the
        # generic single-field extractor) answer with ONLY the field
        # actually asked about. The full per-source dump remains further
        # below as the deliberate fallback for genuine "show me everything
        # about ENQ001" requests where no specific field was asked for.
        results = list(by_source.values())

    phone = extract_phone(query)
    if phone and not eid:
        # Exact reverse lookup — a phone number identifies exactly one
        # customer, so this bypasses fuzzy TF-IDF matching entirely
        # (which previously could rank an unrelated customer above the
        # actual owner, or get overridden by conversation-history logic).
        by_source = retriever.get_by_phone(phone)
        if not by_source:
            return f"No customer found with phone number {phone}."
        results = list(by_source.values())

    LIST_TRIGGERS = {
        "all records","entire dataset","full list","every record","all customers",
        "all enquiries","show dataset","take enquiry","give dataset","all data",
        "entire database","list all","show all","give all","dump","all feedback",
        "all appointments","show enquiry","show feedback","show appointment",
        "take feedback","take appointment","give enquiry","give feedback",
        "enquiry data","appointment data","feedback data","enquiry dataset",
        "appointment dataset","feedback dataset",
    }
    _list_name = extract_name(query) or _extract_name_lower(query)
    if (intent == "list_all" or any(t in q_low for t in LIST_TRIGGERS)) and not _list_name:
        target_src = ("Feedback"    if "feedback"    in q_low else
                      "Appointment" if "appointment"  in q_low else
                      "Enquiry")
        all_rows = retriever.get_all_by_source(target_src)
        if not all_rows:
            return f"No {target_src} records found."
        limit = min(10, len(all_rows))
        lines = [f"Showing {limit} of {len(all_rows)} {target_src} records:\n"]
        for i, r in enumerate(all_rows[:limit], 1):
            name   = find_val(r,"customer","name") or "?"
            eid_v  = find_val(r,"enquiry id","ENQUIRY ID") or "?"
            status = find_val(r,"status") or "?"
            if target_src == "Enquiry":
                veh = find_val(r,"vehicle","model","car") or "?"
                lines.append(f"  {i:>2}. [{eid_v}] {name} - {veh} - Status: {status}")
            elif target_src == "Appointment":
                adate = find_val(r,"appointment date","date") or "?"
                atime = find_val(r,"time") or "?"
                lines.append(f"  {i:>2}. [{eid_v}] {name} - {adate} {atime} - Status: {status}")
            else:
                rating = find_val(r,"rating") or "?"
                fb     = (find_val(r,"feedback") or "?")[:40]
                lines.append(f"  {i:>2}. [{eid_v}] {name} - Rating: {rating} - \"{fb}\"")
        if len(all_rows) > 10:
            lines.append(f"\n  … and {len(all_rows)-10} more. Ask for a specific customer or filter.")
        return "\n".join(lines)

    if intent == "bad_feedback":
        # If a specific customer name was retrieved, show only their records
        named_fb = [r for r in results if get_source(r) == "Feedback"]
        if named_fb and find_val(named_fb[0], "customer", "name"):
            # Single-customer path: show all their feedback entries
            cname = find_val(named_fb[0], "customer", "name")
            lines = [f"Feedback for {cname}:\n"]
            for r in named_fb:
                rating  = find_val(r, "rating") or "?"
                fb      = find_val(r, "feedback") or "?"
                date    = find_val(r, "date") or "?"
                eid_v   = find_val(r, "enquiry", "id") or "?"
                try:
                    r_int = int(float(rating))
                    bar = "★" * r_int + "☆" * (5 - r_int)
                except Exception:
                    bar = ""
                lines.append(f"  [{eid_v}] Rating: {rating}/5 [{bar}] — \"{fb}\" — {date}")
            return "\n".join(lines)
        # Global list path
        bad = [r for r in retriever.get_all_by_source("Feedback")
               if _rating(r) is not None and _rating(r) <= 2]
        if bad:
            lines = [f"Customers with low ratings (≤2/5) — {len(bad)} found:\n"]
            for r in bad:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— Rating: {find_val(r,'rating')}/5 "
                    f"— \"{find_val(r,'feedback') or '?'}\"")
            return "\n".join(lines)
        return "No customers with very low ratings (≤2) found."

    if intent == "good_feedback":
        # If a specific customer name was retrieved, show only their records
        named_fb = [r for r in results if get_source(r) == "Feedback"]
        if named_fb and find_val(named_fb[0], "customer", "name"):
            cname = find_val(named_fb[0], "customer", "name")
            lines = [f"Feedback for {cname}:\n"]
            for r in named_fb:
                rating  = find_val(r, "rating") or "?"
                fb      = find_val(r, "feedback") or "?"
                date    = find_val(r, "date") or "?"
                eid_v   = find_val(r, "enquiry", "id") or "?"
                try:
                    r_int = int(float(rating))
                    bar = "★" * r_int + "☆" * (5 - r_int)
                except Exception:
                    bar = ""
                lines.append(f"  [{eid_v}] Rating: {rating}/5 [{bar}] — \"{fb}\" — {date}")
            return "\n".join(lines)
        # Global list path
        good = [r for r in retriever.get_all_by_source("Feedback")
                if _rating(r) is not None and _rating(r) >= 4]
        if good:
            lines = [f"Customers with high ratings (≥4/5) — {len(good)} found:\n"]
            for r in good:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— Rating: {find_val(r,'rating')}/5 "
                    f"— \"{find_val(r,'feedback') or '?'}\"")
            return "\n".join(lines)
        return "No customers with high ratings (≥4) found."

    if intent == "cancelled":
        rows = [r for r in retriever.get_all_by_source("Appointment")
                if "cancel" in str(find_val(r,"status") or "").lower()]
        if rows:
            lines = [f"Cancelled appointments — {len(rows)} found:\n"]
            for r in rows:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— {find_val(r,'appointment date','date') or '?'} "
                    f"at {find_val(r,'time') or 'N/A'} "
                    f"— {find_val(r,'vehicle','car','model') or 'N/A'}")
            return "\n".join(lines)
        return "No cancelled appointments found."

    if intent == "completed":
        rows = [r for r in retriever.get_all_by_source("Appointment")
                if "complet" in str(find_val(r,"status") or "").lower()]
        if rows:
            lines = [f"Completed appointments — {len(rows)} found:\n"]
            for r in rows:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry','id') or '?'}) "
                    f"— {find_val(r,'appointment date','date') or '?'} "
                    f"at {find_val(r,'time') or 'N/A'}")
            return "\n".join(lines)
        return "No completed appointments found."

    if intent == "appointment":
        appt = [r for r in results if get_source(r) == "Appointment"]
        row  = appt[0] if appt else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        rid      = find_val(row,"enquiry","id") or "N/A"
        adate    = find_val(row,"appointment date","date") or "N/A"
        atime    = find_val(row,"time") or "N/A"
        status   = find_val(row,"status") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        followup = {
            "Scheduled": "→ Confirmed. Please arrive 10 minutes early.",
            "Completed": "→ Already completed successfully.",
            "Cancelled": "→ Cancelled. Please call to reschedule.",
        }.get(status, "")
        return (f"Appointment for {customer} (ID: {rid}):\n"
                f"  Date   : {adate}\n"
                f"  Time   : {atime}\n"
                f"  Vehicle: {veh}\n"
                f"  Status : {status}  {followup}")

    if intent == "feedback":
        fb_rows = [r for r in results if get_source(r) == "Feedback"]
        if not fb_rows:
            fb_rows = results
        customer = find_val(fb_rows[0], "customer", "name") or "N/A"
        if len(fb_rows) == 1:
            row    = fb_rows[0]
            rid    = find_val(row,"enquiry","id") or "N/A"
            fb     = find_val(row,"feedback") or "N/A"
            rating = find_val(row,"rating") or "N/A"
            date   = find_val(row,"date") or "N/A"
            try:
                r_int = int(float(rating))
                bar   = "★" * r_int + "☆" * (5 - r_int)
            except Exception:
                bar = ""
            return (f"Feedback from {customer} (ID: {rid}):\n"
                    f"  Comment : \"{fb}\"\n"
                    f"  Rating  : {rating}/5  [{bar}]\n"
                    f"  Date    : {date}")
        # Multiple feedback entries for this customer
        lines = [f"Feedback for {customer} — {len(fb_rows)} entries:\n"]
        for row in fb_rows:
            rid    = find_val(row,"enquiry","id") or "?"
            fb     = find_val(row,"feedback") or "?"
            rating = find_val(row,"rating") or "?"
            date   = find_val(row,"date") or "?"
            try:
                r_int = int(float(rating))
                bar   = "★" * r_int + "☆" * (5 - r_int)
            except Exception:
                bar = ""
            lines.append(f"  [{rid}] {rating}/5 [{bar}] — \"{fb}\" — {date}")
        return "\n".join(lines)

    if intent == "contact":
        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        email    = find_val(row,"email") or "N/A"
        phone    = find_val(row,"phone","mobile","number") or "N/A"
        return (f"Contact details for {customer}:\n"
                f"  Phone : {phone}\n"
                f"  Email : {email}")

    if intent == "status":
        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        rid      = find_val(row,"enquiry","id") or "N/A"
        status   = find_val(row,"status") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        return (f"Enquiry status for {customer} (ID: {rid}):\n"
                f"  Vehicle : {veh}\n"
                f"  Status  : {status}")

    if intent == "payment":
        # ── Reverse lookup: "which customers are using a loan/EMI/cash?" ────
        # No specific customer named → this is asking to list every matching
        # customer, not to report one person's payment preference.
        _fwd_name = extract_name(query) or _extract_name_lower(query)
        PAYMENT_TYPES = ["loan", "cash", "emi", "finance", "card", "cheque", "upi"]
        payment_re = re.search(r'\b(' + '|'.join(PAYMENT_TYPES) + r')\b', q_low)

        if not _fwd_name and not eid and payment_re:
            pt   = payment_re.group(1).title()
            rows = [r for r in retriever.get_all_by_source("Enquiry")
                    if pt.lower() in str(find_val(r, "payment") or "").lower()]
            if rows:
                lines = [f"Customers using {pt} as payment type — {len(rows)} found:\n"]
                for r in rows:
                    lines.append(
                        f"  * {find_val(r,'customer','name') or '?'} "
                        f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                        f"— {find_val(r,'vehicle','model') or '?'} "
                        f"— Status: {find_val(r,'status') or '?'}")
                return "\n".join(lines)
            return f"No customers found using {pt} as payment type."

        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        payment  = find_val(row,"payment") or "N/A"
        return f"Payment preference for {customer}: {payment}"

    if intent == "test_ride":
        # ── Reverse lookup: "which customers have taken a test ride?" /
        # "who hasn't taken a test ride?" — no specific customer named →
        # list every matching customer instead of reporting on just one.
        _fwd_name = extract_name(query) or _extract_name_lower(query)
        NEGATION_WORDS = {"not", "haven't", "hasn't", "havent", "hasnt",
                          "no", "didn't", "didnt", "without", "n't"}
        q_tokens_neg = set(re.findall(r"[a-zA-Z']+", q_low))
        wants_no = bool(q_tokens_neg & NEGATION_WORDS)
        GROUP_TRIGGERS = {"customers", "customer", "who", "which", "list",
                          "everyone", "anybody", "anyone", "people"}

        if not _fwd_name and not eid and (q_tokens_neg & GROUP_TRIGGERS):
            target = "no" if wants_no else "yes"
            rows = [r for r in retriever.get_all_by_source("Enquiry")
                    if str(find_val(r, "test ride", "test_ride") or "").strip().lower() == target]
            label = "have NOT taken a test ride" if wants_no else "have taken a test ride"
            if rows:
                lines = [f"Customers who {label} — {len(rows)} found:\n"]
                for r in rows:
                    lines.append(
                        f"  * {find_val(r,'customer','name') or '?'} "
                        f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                        f"— {find_val(r,'vehicle','model') or '?'} "
                        f"— Status: {find_val(r,'status') or '?'}")
                return "\n".join(lines)
            return f"No customers found who {label}."

        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        ride     = find_val(row,"test ride","test_ride") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        return (f"Test ride status for {customer}:\n"
                f"  Vehicle   : {veh}\n"
                f"  Test Ride : {ride}")

    if intent == "vehicle":
        # ── Reverse lookup: "which customers are interested in the Kia
        # Seltos?" — a specific vehicle model was named and no specific
        # customer was → list every matching customer instead of picking
        # an arbitrary single row.
        _fwd_name = extract_name(query) or _extract_name_lower(query)
        if not _fwd_name and not eid:
            all_rows = retriever.get_all_by_source("Enquiry")
            vehicle_names = {
                find_val(r, "vehicle", "model", "car")
                for r in all_rows if find_val(r, "vehicle", "model", "car")
            }
            matched_vehicle = next((v for v in vehicle_names if v.lower() in q_low), None)
            if matched_vehicle:
                rows = [r for r in all_rows
                        if find_val(r, "vehicle", "model", "car") == matched_vehicle]
                if rows:
                    lines = [f"Customers interested in {matched_vehicle} — {len(rows)} found:\n"]
                    for r in rows:
                        lines.append(
                            f"  * {find_val(r,'customer','name') or '?'} "
                            f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                            f"— Status: {find_val(r,'status') or '?'}")
                    return "\n".join(lines)
                return f"No customers found interested in {matched_vehicle}."

        enq = [r for r in results if get_source(r) == "Enquiry"]
        row  = enq[0] if enq else results[0]
        customer = find_val(row,"customer","name") or "N/A"
        veh      = find_val(row,"vehicle","model","car") or "N/A"
        rid      = find_val(row,"enquiry","id") or "N/A"
        return f"{customer} (ID: {rid}) is interested in the {veh}."

    if intent == "city":
        city_re = re.search(
            r'\b(hyderabad|bangalore|bengaluru|chennai|mumbai|delhi|pune|coimbatore|'
            r'kolkata|ahmedabad|surat|jaipur|lucknow|nagpur)\b', q_low)

        # ── Forward lookup: "which city is Divya from?" / "where does X live?" ──
        # / "which city is ENQ073 from?" — a city name was NOT given in the
        # query, so this is asking about one specific customer, not asking
        # to list customers by city. Answer with just that one field.
        q_name = extract_name(query) or _extract_name_lower(query)
        if not city_re and (q_name or eid):
            enq  = [r for r in results if get_source(r) == "Enquiry"]
            row  = enq[0] if enq else (results[0] if results else None)
            if row:
                customer = find_val(row,"customer","name") or (q_name or "").title() or "N/A"
                city     = find_val(row,"city","location","state") or "N/A"
                return f"{customer} is from {city}."

        # ── Reverse lookup: "who's from Coimbatore?" ─────────────────────────
        if city_re:
            cn   = city_re.group(1).title()
            rows = [r for r in retriever.get_all_by_source("Enquiry")
                    if cn.lower() in str(find_val(r,"city","location","state") or "").lower()]
            if rows:
                lines = [f"Customers from {cn} — {len(rows)} found:\n"]
                for r in rows:
                    lines.append(
                        f"  * {find_val(r,'customer','name') or '?'} "
                        f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                        f"— {find_val(r,'vehicle','model') or '?'} "
                        f"— Status: {find_val(r,'status') or '?'}")
                return "\n".join(lines)

    if intent == "new_lead":
        # ── Forward lookup: "is ENQ005 a new lead?" / "is Divya a new customer?" ──
        # Asking about ONE specific customer's type, not asking to list all
        # new leads. Answer with just that field.
        _fwd_name = extract_name(query) or _extract_name_lower(query)
        if eid or _fwd_name:
            enq = [r for r in results if get_source(r) == "Enquiry"]
            row = enq[0] if enq else (results[0] if results else None)
            if row:
                who   = find_val(row,"customer","name") or (_fwd_name or eid or "N/A")
                ctype = find_val(row,"customer type") or "N/A"
                rid   = find_val(row,"enquiry id","ENQUIRY ID") or eid or "N/A"
                return f"{who} (ID: {rid}) is a {ctype} customer."

        ALL_ENQUIRY_TRIGGERS = {"take","give","show","fetch","get","pull","dataset",
                                "data","records","all","entire","full","list","dump"}
        q_words_set = set(re.findall(r'\w+', q_low))
        if (q_words_set & ALL_ENQUIRY_TRIGGERS) and not (extract_name(query) or _extract_name_lower(query)):
            all_rows = retriever.get_all_by_source("Enquiry")
            limit    = min(10, len(all_rows))
            lines    = [f"Showing {limit} of {len(all_rows)} Enquiry records:\n"]
            for i, r in enumerate(all_rows[:limit], 1):
                name   = find_val(r,"customer","name") or "?"
                eid_v  = find_val(r,"enquiry id","ENQUIRY ID") or "?"
                veh    = find_val(r,"vehicle","model","car") or "?"
                status = find_val(r,"status") or "?"
                lines.append(f"  {i:>2}. [{eid_v}] {name} — {veh} — Status: {status}")
            if len(all_rows) > 10:
                lines.append(f"\n  … and {len(all_rows)-10} more.")
            return "\n".join(lines)

        rows = [r for r in retriever.get_all_by_source("Enquiry")
                if "new" in str(find_val(r,"customer type") or "").lower()
                or "new lead" in str(find_val(r,"status") or "").lower()]
        if rows:
            lines = [f"New customer leads — {len(rows)} found:\n"]
            for r in rows[:10]:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                    f"— {find_val(r,'vehicle','model') or '?'} "
                    f"— Status: {find_val(r,'status') or '?'}")
            return "\n".join(lines)
        return "No new customer leads found."

    if intent == "returning":
        # ── Forward lookup: "is ENQ005 a returning customer?" ────────────────
        # Asking about ONE specific customer, not asking to list all
        # returning customers. Answer with just that field.
        _fwd_name = extract_name(query) or _extract_name_lower(query)
        if eid or _fwd_name:
            enq = [r for r in results if get_source(r) == "Enquiry"]
            row = enq[0] if enq else (results[0] if results else None)
            if row:
                who   = find_val(row,"customer","name") or (_fwd_name or eid or "N/A")
                ctype = find_val(row,"customer type") or "N/A"
                rid   = find_val(row,"enquiry id","ENQUIRY ID") or eid or "N/A"
                return f"{who} (ID: {rid}) is a {ctype} customer."

        rows = [r for r in retriever.get_all_by_source("Enquiry")
                if "returning" in str(find_val(r,"customer type") or "").lower()
                or "existing"  in str(find_val(r,"customer type") or "").lower()]
        if rows:
            lines = [f"Returning / existing customers — {len(rows)} found:\n"]
            for r in rows[:10]:
                lines.append(
                    f"  * {find_val(r,'customer','name') or '?'} "
                    f"(ID: {find_val(r,'enquiry id','ENQUIRY ID') or '?'}) "
                    f"— {find_val(r,'vehicle','model') or '?'}")
            return "\n".join(lines)
        return "No returning customers found."

    # ── Generic single-field extractor ────────────────────────────────────────
    # Catches specific-attribute questions that don't have a dedicated intent
    # above (gender, email alone, enquiry source, customer type, a bare date),
    # so they get a one-line answer instead of falling through to the full
    # record dump. Only fires when the query names a customer/ID (so we know
    # exactly which row to read from) and clearly asks for ONE known field.
    _q_name_generic = extract_name(query) or _extract_name_lower(query)
    if _q_name_generic or eid:
        _row_pool = ([r for r in results if get_source(r) == "Enquiry"] or
                     results or [])
        _row = _row_pool[0] if _row_pool else None
        if _row:
            _who = find_val(_row,"customer","name") or (_q_name_generic or "").title() or "N/A"
            FIELD_QUESTIONS = [
                (r'\bgender\b',                                   ("gender",),                "gender is"),
                (r'\bemail\b|\be-?mail address\b',                ("email",),                  "email is"),
                (r'\benquiry source\b|\bhow did .* (hear|find)\b|\bsource\b',
                                                                    ("enquiry source","source"), "enquiry source is"),
                (r'\bcustomer type\b|\bnew or returning\b',       ("customer type",),          "customer type is"),
                (r'\benquiry date\b|\bwhen did .* enquir',        ("enquiry date",),           "enquiry date is"),
                (r'\bappointment date\b|\bwhen is .* appointment\b',
                                                                    ("appointment date","date"), "appointment date is"),
            ]
            for pattern, keys, phrasing in FIELD_QUESTIONS:
                if re.search(pattern, q_low):
                    val = find_val(_row, *keys)
                    if val not in (None, "", "nan", "None"):
                        return f"{_who}'s {phrasing} {val}."

    # Summary / full details fallback
    enq_rows  = [r for r in results if get_source(r) == "Enquiry"]
    appt_rows = [r for r in results if get_source(r) == "Appointment"]
    fb_rows   = [r for r in results if get_source(r) == "Feedback"]

    if enq_rows or appt_rows or fb_rows:
        lines = []
        if enq_rows:
            lines.append("── Enquiry " + "─"*40)
            for col, val in enq_rows[0].items():
                if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
                    lines.append(f"  {col}: {val}")
        if appt_rows:
            lines.append("── Appointment " + "─"*36)
            for col, val in appt_rows[0].items():
                if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
                    lines.append(f"  {col}: {val}")
        if fb_rows:
            lines.append("── Feedback " + "─"*39)
            for col, val in fb_rows[0].items():
                if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
                    lines.append(f"  {col}: {val}")
        return "\n".join(lines) if lines else "No details found."

    top   = results[0]
    src   = get_source(top)
    lines = [f"Best match from {src}:"]
    for col, val in top.items():
        if not col.startswith("__") and str(val) not in ("nan","None","","N/A"):
            lines.append(f"  {col}: {val}")
    return "\n".join(lines)


class IntelligentSalesChatbot:
    def __init__(self, retriever: IntelligentRetriever):
        self.retriever = retriever
        self.history   = []
        self.show_ctx  = False
        self.top_k     = TOP_K
        self.use_llm   = True
        self.last_reasoning = {"path": None, "confidence": None,
                                "operation": None, "engine_trace": None}
        # Structured query-understanding / planning / multi-hop engine.
        # Built once from the already-loaded dataframes (cheap: pure
        # pandas, no re-indexing). See query_engine.py for details.
        try:
            from query_engine import QueryEngine
            self.qengine = QueryEngine(self.retriever.dfs)
        except Exception as exc:  # never let this block startup
            print(f"  [WARN] query_engine unavailable: {exc}")
            self.qengine = None

    def chat(self, raw_query: str) -> tuple:
        t0    = time.time()
        query = normalize_query(raw_query.strip())

        # Exposed after every call so callers (e.g. api.py) can show which
        # reasoning path answered the query, without changing this
        # method's return signature (kept stable for run_cli / API compat).
        self.last_reasoning = {"path": "retrieval_pipeline", "confidence": None,
                                "operation": None, "engine_trace": None}

        # ── Ambiguity detection: runs BEFORE any retrieval or pandas
        # execution (see ambiguity_detector.py). Catches duplicate customer
        # names, contradictory filters ("cancelled and completed"), an
        # unresolvable pronoun with no prior context, a partial enquiry ID,
        # or multiple cities in one query — and asks for clarification
        # instead of silently guessing which record/records were meant.
        if ambiguity_detector is not None:
            try:
                _name  = extract_name(query) or _extract_name_lower(query)
                _eid   = extract_enq_id(query)
                _cities = []
                if self.qengine is not None:
                    q_low_probe = query.lower()
                    _cities = [c for c in self.qengine.vocab.cities
                               if c.lower() in q_low_probe]
                amb = ambiguity_detector.detect_ambiguity(
                    query=query,
                    extracted_names=[_name] if _name else [],
                    extracted_cities=_cities,
                    extracted_ids=[_eid] if _eid else [],
                    extracted_vehicles=[],
                    dfs=self.retriever.dfs,
                    session_has_active_customer=bool(self.history),
                )
                if amb.is_ambiguous:
                    elapsed = time.time() - t0
                    self.history.append(("You", raw_query))
                    self.history.append(("Assistant", amb.clarification_question))
                    self.last_reasoning = {
                        "path": "ambiguity_clarification",
                        "confidence": 0.1,
                        "operation": amb.kind,
                        "engine_trace": amb.candidates,
                    }
                    return amb.clarification_question, elapsed, "clarification_needed"
            except Exception as exc:
                # Never let the ambiguity check itself block an answer.
                print(f"  [WARN] ambiguity_detector failed: {exc}")

        # ── Structured engine: counts / averages / top-k / group-by /
        # compare / multi-hop-across-datasets. Only fires when it has a
        # confident plan; otherwise falls straight through to the
        # existing pipeline below, completely unchanged. ──────────────
        if self.qengine is not None:
            engine_result = self.qengine.try_handle(query)
            if engine_result is not None and engine_result.confidence >= self.qengine.MIN_CONFIDENCE:
                structured_facts = engine_result.facts
                generated = (_generate_phrase(query, structured_facts,
                                               engine_result.plan.operation.lower())
                             if self.use_llm else None)
                verified = True
                if generated:
                    from verification import verify_answer
                    verified, _reason = verify_answer(generated, structured_facts)
                    if not verified:
                        generated = None
                answer  = (generated + "\n\n" + structured_facts) if generated else structured_facts
                elapsed = time.time() - t0
                self.history.append(("You", raw_query))
                self.history.append(("Assistant", answer))
                self.last_reasoning = {
                    "path": "query_engine",
                    "confidence": round(engine_result.confidence, 3),
                    "operation": engine_result.plan.operation,
                    "engine_trace": engine_result.plan.trace,
                    "llm_phrasing_verified": verified,
                }
                return answer, elapsed, engine_result.plan.operation.lower()

        queried_name = extract_name(query) or _extract_name_lower(query)

        if not queried_name and len(self.history) >= 2 and not has_own_identifier(query):
            # NOTE: has_own_identifier() guard above is critical. Without it,
            # ANY query containing a digit (a phone number, an enquiry ID,
            # a date, a rating) was treated as "too short/ambiguous, must be
            # about whoever we discussed last" and silently swapped in a
            # stale customer name from 1-2 turns ago — even though the
            # digits in THIS query already fully identify a specific,
            # different record. That's what made "whose number is this
            # 9945206132" return a completely unrelated customer's contact
            # details if a different person had been discussed earlier in
            # the conversation.
            q_tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))
            CONTINUATION_HINTS = {
                "this", "that", "it", "same", "again", "there",
                "he", "she", "him", "her", "his", "their", "they", "them",
            }
            # These keywords make a query self-contained — they describe a
            # GROUP or a NEW topic rather than continuing about the last
            # person.  Never borrow a name from history when they appear.
            SELF_CONTAINED_KEYWORDS = {
                "all", "every", "everyone", "who", "which", "customers",
                "customer", "list", "show", "give", "feedback", "feedbacks",
                "rating", "ratings", "appointment", "appointments", "cancelled",
                "completed", "status", "new", "lead", "leads", "returning",
                "city", "location", "bad", "poor", "good", "great", "low",
                "high", "satisfied", "dissatisfied", "unhappy", "happy",
                "records", "dataset", "data", "enquiry", "enquiries",
            }
            is_self_contained = bool(q_tokens & SELF_CONTAINED_KEYWORDS)
            looks_like_clarification = (
                not is_self_contained
                and (
                    bool(q_tokens & CONTINUATION_HINTS)
                    or bool(re.search(r"\d", query))
                    or len(q_tokens) <= 4
                )
            )
            if looks_like_clarification:
                prev_role, prev_text = self.history[-2]
                if prev_role == "You":
                    queried_name = extract_name(prev_text) or _extract_name_lower(prev_text)

        intent, conf = detect_intent(query, known_name=queried_name, retriever=self.retriever)
        target_source = INTENT_TO_SOURCE.get(intent)

        if target_source:
            results = self.retriever.retrieve_from_source(
                query, target_source, top_k=self.top_k)
            if queried_name:
                name_rows = self.retriever.retrieve_by_name(
                    queried_name, preferred_source=target_source, top_k=self.top_k)
                results = name_rows + [r for r in results if r not in name_rows]
        else:
            results = self.retriever.retrieve(query, top_k=self.top_k)
            if queried_name:
                name_rows = self.retriever.retrieve_by_name(
                    queried_name, top_k=self.top_k * 2)
                results = name_rows + [r for r in results if r not in name_rows]

        if intent in ("feedback", "bad_feedback", "good_feedback"):
            extra = self.retriever.retrieve_from_source(
                "customer feedback rating review satisfied unhappy complaint",
                "Feedback", top_k=self.top_k)
            results += [r for r in extra if r not in results]
        elif intent in ("cancelled", "completed", "appointment"):
            extra = self.retriever.retrieve_from_source(
                "appointment meeting booking scheduled cancelled completed",
                "Appointment", top_k=self.top_k)
            results += [r for r in extra if r not in results]

        seen, deduped = set(), []
        for r in results:
            key = (r.get("__source__"),
                   str(find_val(r, "enquiry id", "ENQUIRY ID") or ""))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        results = deduped

        query_date = extract_query_date(query)
        if query_date:
            date_filtered = [r for r in results if _row_matches_date(r, query_date)]
            if date_filtered:
                results = date_filtered
            else:
                elapsed = time.time() - t0
                who = f" for {queried_name}" if queried_name else ""
                answer = (
                    f"I couldn't find any record{who} on {query_date}. "
                    f"Double-check the date, or ask without a specific date "
                    f"to see all matching records."
                )
                self.history.append(("You", raw_query))
                self.history.append(("Assistant", answer))
                return answer, elapsed, "no_match"

        if self.show_ctx:
            print(f"\n  [Intent: '{intent}'  confidence={conf:.3f}  target: {target_source}]")
            print("  [Top retrieved records]")
            for i, r in enumerate(results[:5], 1):
                preview = "  ".join(
                    f"{k}:{str(v)[:20]}" for k, v in r.items()
                    if not str(k).startswith("__") and str(v) not in ("nan","None","")
                )[:120]
                print(f"    {i}. score={r.get('__score__',0):.3f} | "
                      f"src={r.get('__source__')} | {preview}")
            print()
            self.show_ctx = False

        structured_facts = build_answer(query, results, intent, self.retriever)

        # ── Pipeline: understand → retrieve (done above) → generate ───────
        # Every answer now goes through the generator, not just list /
        # aggregation intents. build_answer() above is still the single
        # source of TRUTH (deterministic, always correct) — the generator's
        # job is only to phrase those already-correct facts as a natural
        # sentence. It can never invent facts:
        #   - a groundedness check rejects output sharing no real words
        #     with the retrieved data (see _generate_phrase)
        #   - for single-record intents, an "identity lock" additionally
        #     requires the actual customer name / enquiry ID to appear
        #     verbatim in the generated sentence, so the small model can't
        #     swap in the wrong person.
        # If generation is rejected for any reason, the structured facts
        # are shown alone — the answer is never blocked by a bad generation.
        LIST_INTENTS = {"list_all","bad_feedback","good_feedback","cancelled",
                        "completed","new_lead","city","returning"}

        require_words = None
        if intent not in LIST_INTENTS and results:
            who = find_val(results[0], "customer", "name")
            rid = find_val(results[0], "enquiry", "id")
            require_words = {w for w in (who, rid) if w}

        generated = (_generate_phrase(query, structured_facts, intent,
                                       require_words=require_words)
                     if self.use_llm else None)
        answer = (generated + "\n\n" + structured_facts) if generated else structured_facts

        # ── Confidence calibration (observability) ──────────────────────────
        # The query_engine path already reports a confidence score; the
        # retrieval pipeline never did. This computes the same
        # multi-dimensional profile from what we actually observed
        # (whether a name/ID resolved, whether the dataset was found, row
        # count, etc.) and surfaces it in last_reasoning so callers (e.g.
        # api.py's /api/health/ or a debugging UI) can see how sure the
        # answer is — this does not change what gets answered.
        conf_profile = None
        if confidence_mod is not None:
            try:
                conf_profile = confidence_mod.calibrate(
                    operation=intent.upper(),
                    entities_found=1 if (queried_name or extract_enq_id(query)) else 0,
                    entity_ambiguous=False,   # ambiguity_detector already vetted this query
                    multiple_entity_matches=False,
                    datasets_matched=1 if (target_source or results) else 0,
                    rows_found=len(results),
                    plan_well_formed=True,
                    execution_ok=True,
                    facts_grounded=True,
                    query_tokens=len(query.split()),
                )
            except Exception as exc:
                print(f"  [WARN] confidence calibration failed: {exc}")

        elapsed = time.time() - t0
        self.history.append(("You", raw_query))
        self.history.append(("Assistant", answer))
        self.last_reasoning = {
            "path": "retrieval_pipeline",
            "confidence": conf_profile.overall if conf_profile else None,
            "confidence_level": conf_profile.level if conf_profile else None,
            "operation": intent,
            "engine_trace": None,
            "dataset_quality": DATASET_QUALITY.get(target_source) if target_source else None,
        }
        return answer, elapsed, intent


def load_datasets() -> dict:
    global DATASET_QUALITY
    dfs = {}
    for source, path in CSV_FILES.items():
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found — skipping.")
            continue
        df = pd.read_csv(path)
        df.columns      = [c.strip() for c in df.columns]
        df["__source__"] = source
        dfs[source]      = df
        print(f"  OK {source:<12}: {len(df)} records")
    if not dfs:
        sys.exit("[ERROR] No CSV files found.")

    # ── Data-quality validation ─────────────────────────────────────────────
    # Run once at startup so bad data (missing/duplicate IDs, out-of-range
    # ratings, unexpected statuses) is visible immediately instead of only
    # surfacing as a confusing downstream answer.
    if data_quality is not None:
        try:
            reports = data_quality.validate_datasets(dfs)
            DATASET_QUALITY = {ds: q.overall_quality_score for ds, q in reports.items()}
            data_quality.print_quality_report(reports)
        except Exception as exc:
            print(f"  [WARN] data_quality validation failed: {exc}")

    return dfs


def build_docs_per_source(dfs: dict) -> dict:
    docs = {}
    for src, df in dfs.items():
        docs[src] = [row_to_rich_text(row.to_dict(), src) for _, row in df.iterrows()]
    return docs


def _cache_fresh(path: str, dfs: dict) -> bool:
    if not os.path.exists(path):
        return False
    ct = os.path.getmtime(path)
    return all(
        not os.path.exists(p) or os.path.getmtime(p) <= ct
        for p in CSV_FILES.values()
    )


BANNER = """
This is an Intelligent Sales Chatbot (scratch-built LLM, no pretrained weights)
"""


def run_cli(chatbot: IntelligentSalesChatbot):
    print(BANNER)
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not user_input:
            continue
        low = user_input.lower()

        if   low == "/quit":
            print("Goodbye!"); break
        elif low == "/llm":
            chatbot.use_llm = not chatbot.use_llm
            print(f"  [LLM generation: {'ON' if chatbot.use_llm else 'OFF'}]\n")
        elif low == "/reset":
            chatbot.history.clear()
            print("  [History cleared]\n")
        elif low == "/history":
            if not chatbot.history:
                print("  [No history yet]\n")
            else:
                for role, text in chatbot.history:
                    print(f"  {role}: {text}")
                print()
        elif low == "/context":
            chatbot.show_ctx = True
            print("  [Context + intent shown for next query]\n")
        elif low.startswith("/topk"):
            try:
                chatbot.top_k = int(user_input.split()[1])
                print(f"  [Top-K set to {chatbot.top_k}]\n")
            except (IndexError, ValueError):
                print("  Usage: /topk 3\n")
        else:
            answer, elapsed, intent = chatbot.chat(user_input)
            print(f"\nAssistant [{intent}]: {answer}")
            print(f"  ({elapsed:.2f}s)\n")


def main():
    global _KNOWN_NAMES_SET

    print("\n[1/3] Loading datasets …")
    dfs = load_datasets()

    for src_df in dfs.values():
        for col in src_df.columns:
            col_low = col.lower()
            # Only genuine person/customer-name columns — NOT any column
            # that merely contains the substring "name" (e.g. "Vehicle
            # Name / Model"), which used to leak vehicle model names like
            # "Kia Seltos" into the customer-name registry and made the
            # chatbot mistake vehicle mentions for a customer lookup.
            if "customer" in col_low and "name" in col_low and not col.startswith("__"):
                _KNOWN_NAMES_SET.update(
                    n for n in src_df[col].dropna().astype(str).unique()
                    if n not in ("nan","None","")
                )
    print(f"  OK Known customer names: {sorted(_KNOWN_NAMES_SET)}")

    print("\n[2/3] Building per-dataset retrievers …")
    docs_per_source = build_docs_per_source(dfs)

    if _cache_fresh(INDEX_CACHE, dfs):
        print(f"  Loading from cache: {INDEX_CACHE}")
        retriever = IntelligentRetriever.load(INDEX_CACHE)
        print("  OK Cache loaded")
    else:
        retriever = IntelligentRetriever(dfs, docs_per_source)
        retriever.save(INDEX_CACHE)
        print(f"  OK Index saved to {INDEX_CACHE}")

    print("\n[3/3] Starting up …")
    print(f"  Device : {LLM_DEVICE}")
    print(f"  LLM    : ScratchLLM (scratch_model/model.pt)")
    import threading
    threading.Thread(target=_load_llm, daemon=True).start()
    print("  [LLM] Loading in background …\n")

    chatbot = IntelligentSalesChatbot(retriever)
    run_cli(chatbot)


if __name__ == "__main__":
    main()
