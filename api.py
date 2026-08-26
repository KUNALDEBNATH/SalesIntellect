import os
import sys
import json
import threading

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "__main__")

from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="sales-rag-chatbot-dev-secret-key",
        ALLOWED_HOSTS=["*"],
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "corsheaders",
            "rest_framework",
        ],
        MIDDLEWARE=[
            "corsheaders.middleware.CorsMiddleware",
            "django.middleware.common.CommonMiddleware",
        ],
        CORS_ALLOW_ALL_ORIGINS=True,
        CORS_ALLOW_HEADERS=[
            "accept",
            "accept-encoding",
            "authorization",
            "content-type",
            "dnt",
            "origin",
            "user-agent",
            "x-csrftoken",
            "x-requested-with",
        ],
        ROOT_URLCONF=__name__,
        DATABASES={},
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
        # Allow uploads up to ~20 MB (attachment_handler.py enforces the
        # precise per-type limits: 15 MB documents / 8 MB images).
        DATA_UPLOAD_MAX_MEMORY_SIZE=20 * 1024 * 1024,
        FILE_UPLOAD_MAX_MEMORY_SIZE=20 * 1024 * 1024,
    )

import django
django.setup()

from django.urls import path
from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse, Http404
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

try:
    from test import (
        IntelligentSalesChatbot,
        IntelligentRetriever,
        load_datasets,
        build_docs_per_source,
        _load_llm,
        _llm_ready,
        CSV_FILES,
        _cache_fresh,
        INDEX_CACHE,
        _KNOWN_NAMES_SET,
        get_dataset_quality,
    )
    IMPORT_OK = True
except ImportError as e:
    print(f"[WARNING] Could not import test.py: {e}")
    IMPORT_OK = False

# ── New: file attachment support (document + image analysis) ────────────────
import domain_guard
import attachment_handler
import bot_identity
from attachment_handler import handle_attachment, AttachmentError

_chatbot: "IntelligentSalesChatbot | None" = None
_init_error: str = ""
_datasets_loaded: list = []

# ── Per-session conversation state ───────────────────────────────────────────
# `_chatbot` (retriever + query engine + scratch LLM) is one expensive,
# shared, read-only object — that part is fine to share across every caller.
# Its `.history` list is NOT: previously every request reused the same
# global chatbot instance's history, `domain_guard`'s "last turn was
# sales" flag, and `attachment_handler`'s stored document, so two different
# callers hitting this API at the same time would see and influence each
# other's conversation. Each session now gets its own history list (and
# domain_guard / attachment_handler are keyed by the same session_id — see
# their per-session updates). Sessions default to "default" so a client
# that never sends a session_id behaves exactly as before (single shared
# conversation), while a client that does send one gets real isolation.
_DEFAULT_SESSION = "default"
_session_histories: dict[str, list] = {}
_SESSION_LOCK = threading.Lock()


def _get_session_id(request, body: dict, is_multipart: bool) -> str:
    if is_multipart:
        sid = request.POST.get("session_id")
    else:
        sid = body.get("session_id")
    sid = (sid or request.headers.get("X-Session-Id") or _DEFAULT_SESSION).strip()
    return sid or _DEFAULT_SESSION


def _get_session_history(session_id: str) -> list:
    with _SESSION_LOCK:
        return _session_histories.setdefault(session_id, [])


def _init_chatbot():
    global _chatbot, _init_error, _datasets_loaded, _KNOWN_NAMES_SET

    if not IMPORT_OK:
        _init_error = "test.py could not be imported. Make sure it is in the same directory."
        return

    try:
        print("[API] Loading datasets …")
        dfs = load_datasets()
        _datasets_loaded = list(dfs.keys())

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
                        if n not in ("nan", "None", "")
                    )

        # ── Teach domain_guard the real vocabulary of this dataset ──────
        # Any customer name, city/state, or vehicle model mentioned in a
        # query should count as in-domain, even without a keyword like
        # "customer" or "vehicle" ("list the persons from Chennai",
        # "anything about the Kia Seltos?").
        entity_cols_keywords = ("name", "city", "state", "vehicle", "model")
        for src_df in dfs.values():
            for col in src_df.columns:
                col_low = col.lower()
                if any(k in col_low for k in entity_cols_keywords) and not col_low.startswith("__"):
                    domain_guard.register_known_entities(src_df[col])
        print(f"[API] domain_guard aware of {len(domain_guard._KNOWN_ENTITY_WORDS)} entity words")

        print("[API] Building retrievers …")
        docs_per_source = build_docs_per_source(dfs)

        if _cache_fresh(INDEX_CACHE, dfs):
            print(f"[API] Loading index from cache: {INDEX_CACHE}")
            retriever = IntelligentRetriever.load(INDEX_CACHE)
        else:
            retriever = IntelligentRetriever(dfs, docs_per_source)
            retriever.save(INDEX_CACHE)

        _chatbot = IntelligentSalesChatbot(retriever)

        threading.Thread(target=_load_llm, daemon=True).start()
        print("[API] Chatbot ready. LLM loading in background …")

    except Exception as exc:
        _init_error = str(exc)
        print(f"[API] Initialisation error: {exc}")

_init_chatbot()


def _json_response(data: dict, status: int = 200) -> JsonResponse:
    return JsonResponse(data, status=status, json_dumps_params={"ensure_ascii": False})


def _parse_body(request) -> dict:
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return {}

@method_decorator(csrf_exempt, name="dispatch")
class ChatView(View):
    """
    POST /api/chat/

    Two supported request formats:

    1. JSON (unchanged, existing behaviour):
         Content-Type: application/json
         Body: {"query": "Show ENQ001 details"}

    2. multipart/form-data (NEW — used automatically by the frontend when a
       file is attached):
         fields: query=<text>, file=<uploaded file>

    Returns (both cases): {"answer": "...", "intent": "...", "elapsed": 1.23,
                            "history_length": 4, "filename": "…" (only if a
                            file was processed)}
    """

    def post(self, request):
        if _chatbot is None:
            return _json_response(
                {"error": f"Chatbot not initialised. {_init_error}"}, status=503
            )

        is_multipart = request.content_type.startswith("multipart/form-data")

        if is_multipart:
            query = (request.POST.get("query") or "").strip()
            uploaded_file = request.FILES.get("file")
            body = {}
        else:
            body = _parse_body(request)
            query = (body.get("query") or "").strip()
            uploaded_file = None

        if not query and not uploaded_file:
            return _json_response({"error": "Missing 'query' in request body."}, status=400)

        # ── Session isolation ─────────────────────────────────────────────
        # `_chatbot` (retriever, query engine, scratch LLM) is shared — that
        # part is expensive and read-only, so sharing it is fine. Its
        # conversation history is NOT shared: point `_chatbot.history` at
        # this session's own list before doing anything else, so every
        # history mutation below lands in the right place. domain_guard and
        # attachment_handler are given the same session_id explicitly so
        # their "last turn was sales" flag / stored document also stay
        # scoped to this caller.
        session_id = _get_session_id(request, body, is_multipart)
        _chatbot.history = _get_session_history(session_id)

        # ── Identity questions ("who are you", "who made you", …) ────────
        # Answered directly with a fixed, branded reply — BEFORE the domain
        # check and BEFORE the retrieval/LLM pipeline — so they can't (a)
        # get misrouted through the "short clarification of the previous
        # sales turn" fallback in domain_guard and (b) end up handed to the
        # LLM alongside a few random customer rows, which is what produces
        # hallucinated nonsense like the model inventing schools/people.
        if not uploaded_file and bot_identity.is_identity_question(query):
            answer = bot_identity.identity_answer()
            _chatbot.history.append(("You", query))
            _chatbot.history.append(("Assistant", answer))
            domain_guard.mark_other_turn(session_id)
            attachment_handler.mark_other_turn(session_id)
            return _json_response(
                {
                    "answer": answer,
                    "intent": "identity",
                    "elapsed": 0.0,
                    "history_length": len(_chatbot.history),
                }
            )

        # ── Detect a conversational follow-up about a previously uploaded
        # file BEFORE the domain check, so natural continuations like
        # "now from this take the top two" or "summarize it again" aren't
        # mistaken for an off-topic question just because they use a
        # pronoun instead of the word "document"/"pdf".
        is_context_followup = (
            not uploaded_file and bool(query)
            and attachment_handler.has_stored_document(session_id)
            and attachment_handler.looks_like_context_followup(query)
        )

        # ── Domain restriction: never let unrelated questions reach the LLM ──
        if not is_context_followup and not domain_guard.is_in_domain(
                query, has_attachment=bool(uploaded_file), session_id=session_id):
            attachment_handler.mark_other_turn(session_id)
            domain_guard.mark_other_turn(session_id)
            refusal = domain_guard.refusal_message()
            _chatbot.history.append(("You", query))
            _chatbot.history.append(("Assistant", refusal))
            return _json_response(
                {
                    "answer": refusal,
                    "intent": "out_of_domain",
                    "elapsed": 0.0,
                    "history_length": len(_chatbot.history),
                }
            )

        # ── File attachment path ─────────────────────────────────────────
        if uploaded_file:
            try:
                result = handle_attachment(uploaded_file, query, chatbot=_chatbot,
                                            session_id=session_id)
            except AttachmentError as exc:
                return _json_response({"error": str(exc)}, status=400)
            except Exception as exc:
                return _json_response({"error": f"Attachment processing failed: {exc}"}, status=500)

            _chatbot.history.append(("You", query or f"[Uploaded file: {result['filename']}]"))
            _chatbot.history.append(("Assistant", result["answer"]))
            domain_guard.mark_other_turn(session_id)

            return _json_response(
                {
                    "answer": result["answer"],
                    "intent": result["intent"],
                    "elapsed": result["elapsed"],
                    "history_length": len(_chatbot.history),
                    "filename": result["filename"],
                }
            )

        # ── Follow-up about a previously uploaded file (no re-upload) ────
        # e.g. "explain the pdf", "now from this take the top two",
        # "summarize it again" — anything that keeps talking about the
        # file already in focus, sent as plain text with no new attachment.
        if is_context_followup:
            result = attachment_handler.answer_from_stored_document(
                query, chatbot=_chatbot, session_id=session_id)
            if result:
                _chatbot.history.append(("You", query))
                _chatbot.history.append(("Assistant", result["answer"]))
                domain_guard.mark_other_turn(session_id)
                return _json_response(
                    {
                        "answer": result["answer"],
                        "intent": result["intent"],
                        "elapsed": result["elapsed"],
                        "history_length": len(_chatbot.history),
                        "filename": result["filename"],
                    }
                )
            # No stored document after all — fall through to normal chat.

        # ── Existing text-only path (unchanged) ──────────────────────────
        try:
            answer, elapsed, intent = _chatbot.chat(query)
            attachment_handler.mark_other_turn(session_id)
            domain_guard.mark_sales_turn(session_id)
            response = {
                "answer": answer,
                "intent": intent,
                "elapsed": round(elapsed, 3),
                "history_length": len(_chatbot.history),
            }
            # Additive field — existing clients that don't read it are
            # unaffected. Shows whether the new structured query_engine
            # (counts / averages / multi-hop joins / comparisons), the
            # ambiguity-clarification path, or the original retrieval
            # pipeline produced this answer, plus a confidence score.
            reasoning = getattr(_chatbot, "last_reasoning", None)
            if reasoning:
                response["reasoning"] = reasoning
            return _json_response(response)
        except Exception as exc:
            return _json_response({"error": str(exc)}, status=500)


@method_decorator(csrf_exempt, name="dispatch")
class ResetView(View):
    """
    POST /api/reset/
    Clears this session's conversation history (send {"session_id": "..."}
    to reset a specific session; omit it to reset the "default" session).
    Returns: {"status": "ok"}
    """

    def post(self, request):
        if _chatbot is None:
            return _json_response({"error": "Chatbot not initialised."}, status=503)
        body = _parse_body(request)
        session_id = (body.get("session_id") or
                      request.headers.get("X-Session-Id") or _DEFAULT_SESSION).strip() or _DEFAULT_SESSION
        with _SESSION_LOCK:
            _session_histories[session_id] = []
        attachment_handler.clear_stored_document(session_id)
        domain_guard.mark_other_turn(session_id)
        return _json_response({"status": "ok"})


class HealthView(View):
    """
    GET /api/health/
    Returns current system status.
    """

    def get(self, request):
        import test as _test_mod  

        llm_ready = getattr(_test_mod, "_llm_ready", False)

        try:
            import vision_parser
            vlm_info = vision_parser.vlm_status()
        except Exception:
            vlm_info = {"vlm_ready": False, "vlm_attempted": False}

        query_engine_info = {"ready": False}
        if _chatbot is not None and getattr(_chatbot, "qengine", None) is not None:
            vocab = _chatbot.qengine.vocab
            query_engine_info = {
                "ready": True,
                "known_names": len(vocab.names),
                "known_cities": len(vocab.cities),
                "known_vehicles": len(vocab.vehicles),
                "supported_operations": [
                    "COUNT", "SUM", "AVERAGE", "MAX", "MIN", "TOP_K",
                    "BOTTOM_K", "GROUP_BY", "COMPARE", "MULTI_HOP",
                ],
            }

        try:
            dataset_quality = get_dataset_quality()
        except Exception:
            dataset_quality = {}

        return _json_response(
            {
                "status": "ok" if _chatbot is not None else "error",
                "chatbot_ready": _chatbot is not None,
                "llm_ready": llm_ready,
                "vlm_ready": vlm_info.get("vlm_ready", False),
                "query_engine": query_engine_info,
                "datasets": _datasets_loaded,
                "dataset_quality": dataset_quality,
                "active_sessions": len(_session_histories),
                "init_error": _init_error or None,
                "index_cache": os.path.abspath(INDEX_CACHE) if os.path.exists(INDEX_CACHE) else None,
            }
        )


class SuggestionsView(View):
    """
    GET /api/suggestions/
    Returns example queries to display in the UI.
    """

    SUGGESTIONS = [
        "Show all enquiries",
        "Who gave bad feedback?",
        "All cancelled appointments",
        "Show ENQ001 full details",
        "Returning customers",
        "Customers from Chennai",
        "Who hasn't taken a test ride?",
        "Show me new leads",
        "What's Divya's feedback?",
        "Is Arjun's appointment confirmed?",
        "What car did Sneha enquire about?",
        "Show good feedback customers",
        "All completed appointments",
        "Payment type breakdown",
        # New: structured / multi-hop reasoning (query_engine.py)
        "How many enquiries are from Chennai?",
        "What is the average feedback rating?",
        "Top 5 customers by rating",
        "Which customers gave negative feedback and also cancelled their appointments?",
        "Compare Ananya and Arjun",
        "Group enquiries by city",
    ]

    def get(self, request):
        return _json_response({"suggestions": self.SUGGESTIONS})


class IndexView(View):
    """
    GET /
    Serves index.html from the same directory as api.py.
    """

    def get(self, request):
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
        if not os.path.exists(html_path):
            return HttpResponse(
                "<h2>index.html not found.</h2>"
                "<p>Place <code>index.html</code> in the same directory as <code>api.py</code>.</p>",
                content_type="text/html",
                status=404,
            )
        with open(html_path, "r", encoding="utf-8") as f:
            return HttpResponse(f.read(), content_type="text/html")


def favicon_view(request):
    """Suppress 404 noise for favicon requests."""
    return HttpResponse(status=204)


urlpatterns = [
    path("",                 IndexView.as_view()),
    path("api/chat/",        ChatView.as_view()),
    path("api/reset/",       ResetView.as_view()),
    path("api/health/",      HealthView.as_view()),
    path("api/suggestions/", SuggestionsView.as_view()),
    path("favicon.ico",      favicon_view),
]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from django.core.management import execute_from_command_line

    # Default: runserver 0.0.0.0:8000
    args = sys.argv if len(sys.argv) > 1 else [sys.argv[0], "runserver", "0.0.0.0:8000"]
    execute_from_command_line(args)
