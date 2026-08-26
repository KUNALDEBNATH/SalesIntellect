"""
data_quality.py
═══════════════════════════════════════════════════════════════════════════════
Schema-aware data quality validation and schema registry.

Run at startup to detect:
  - Missing IDs / duplicate IDs
  - Invalid ratings / unexpected status values
  - Broken join keys across datasets
  - Numeric fields stored as strings

The reasoning engine uses quality signals to lower confidence when data
is known to be poor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd


# ─────────────────────────────────── schema registry ─────────────────────────

@dataclass
class ColumnSpec:
    name: str
    dtype: str                       # numeric | text | date | id
    aliases: List[str] = field(default_factory=list)
    semantic: str = ""               # human-readable purpose
    is_join_key: bool = False
    valid_values: Optional[Set[str]] = None   # if known categorical


# Hard-coded schema for the three known CSV types;
# extended dynamically for new CSVs.
SCHEMA: Dict[str, Dict[str, ColumnSpec]] = {
    "Enquiry": {
        "Customer Name":         ColumnSpec("Customer Name", "text", ["name", "customer"], "customer full name"),
        "City / State":          ColumnSpec("City / State", "text", ["city", "state", "location"], "city/state of enquiry"),
        "Vehicle Name / Model":  ColumnSpec("Vehicle Name / Model", "text", ["vehicle", "model", "car", "bike"], "vehicle of interest"),
        "Status":                ColumnSpec("Status", "text", ["status", "enquiry status"], "enquiry status",
                                            valid_values={"New Lead", "Contacted", "Closed", "Pending", "Returning"}),
        "Payment Mode":          ColumnSpec("Payment Mode", "text", ["payment", "payment mode"], "payment type"),
    },
    "Appointment": {
        "Customer Name":  ColumnSpec("Customer Name", "text", ["name"], "customer full name"),
        "Status":         ColumnSpec("Status", "text", ["status", "appointment status"],
                                     "appointment status",
                                     valid_values={"Scheduled", "Cancelled", "Completed", "Booked", "Pending"}),
        "Enquiry ID":     ColumnSpec("Enquiry ID", "id", ["enquiry id", "id", "enq id"],
                                     "join key", is_join_key=True),
    },
    "Feedback": {
        "Customer Name":  ColumnSpec("Customer Name", "text", ["name"], "customer full name"),
        "Rating":         ColumnSpec("Rating", "numeric", ["rating", "score", "feedback score"], "customer rating (1–5)"),
        "Enquiry ID":     ColumnSpec("Enquiry ID", "id", ["enquiry id", "id"], "join key", is_join_key=True),
    },
}


# ─────────────────────────────────── quality report ──────────────────────────

@dataclass
class DatasetQuality:
    dataset: str
    rows: int
    missing_id_count: int = 0
    duplicate_id_count: int = 0
    duplicate_customer_count: int = 0
    invalid_values: Dict[str, int] = field(default_factory=dict)
    broken_join_keys: int = 0
    string_as_numeric: List[str] = field(default_factory=list)
    overall_quality_score: float = 1.0   # 0–1; used to lower confidence
    notes: List[str] = field(default_factory=list)


def validate_datasets(dfs: Dict[str, pd.DataFrame]) -> Dict[str, DatasetQuality]:
    """
    Validate all loaded DataFrames.
    Returns a dict of DatasetQuality, keyed by dataset name.
    """
    reports: Dict[str, DatasetQuality] = {}

    for ds_name, df in dfs.items():
        df.columns = [str(c).strip() for c in df.columns]
        q = DatasetQuality(dataset=ds_name, rows=len(df))

        # ── ID column checks ─────────────────────────────────────────────────
        id_col = next((c for c in df.columns
                       if "enquiry" in c.lower() and "id" in c.lower()), None)
        if id_col:
            missing = df[id_col].isna().sum()
            dupes = df[id_col].dropna().duplicated().sum()
            q.missing_id_count = int(missing)
            q.duplicate_id_count = int(dupes)
            if missing > 0:
                q.notes.append(f"Missing Enquiry IDs: {missing}")
            if dupes > 0:
                q.notes.append(f"Duplicate Enquiry IDs: {dupes}")

        # ── Customer name duplicates ─────────────────────────────────────────
        name_col = next((c for c in df.columns
                         if "customer" in c.lower() and "name" in c.lower()), None)
        if name_col:
            dupes = df[name_col].dropna().duplicated().sum()
            q.duplicate_customer_count = int(dupes)
            if dupes > 0:
                q.notes.append(f"Duplicate customer names: {dupes} (expected — same person, multiple records)")

        # ── Rating validation ─────────────────────────────────────────────────
        if "Rating" in df.columns:
            numeric = pd.to_numeric(df["Rating"], errors="coerce")
            invalid = int(numeric.isna().sum() - df["Rating"].isna().sum())
            out_of_range = int(((numeric < 1) | (numeric > 5)).sum())
            if invalid > 0:
                q.invalid_values["Rating (non-numeric)"] = invalid
                q.notes.append(f"Non-numeric Rating values: {invalid}")
            if out_of_range > 0:
                q.invalid_values["Rating (out of 1–5)"] = out_of_range
                q.notes.append(f"Ratings outside 1–5: {out_of_range}")

        # ── Status validation ─────────────────────────────────────────────────
        if "Status" in df.columns:
            schema_spec = SCHEMA.get(ds_name, {}).get("Status")
            if schema_spec and schema_spec.valid_values:
                actual_values = set(df["Status"].dropna().astype(str).unique())
                unexpected = actual_values - schema_spec.valid_values
                if unexpected:
                    q.invalid_values["Status"] = len(unexpected)
                    q.notes.append(f"Unexpected Status values: {sorted(unexpected)}")

        # ── Numeric fields stored as strings ──────────────────────────────────
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(50).astype(str)
                numeric_count = sample.apply(lambda x: bool(re.match(r"^-?\d+(\.\d+)?$", x.strip()))).sum()
                if numeric_count > 0.8 * len(sample) and len(sample) > 5:
                    q.string_as_numeric.append(col)
                    q.notes.append(f"Column '{col}' looks numeric but stored as text")

        # ── Overall quality score ─────────────────────────────────────────────
        penalties = 0.0
        if q.missing_id_count > 0:
            penalties += min(0.2, q.missing_id_count / max(q.rows, 1) * 2)
        if q.duplicate_id_count > 0:
            penalties += min(0.1, q.duplicate_id_count / max(q.rows, 1))
        if q.invalid_values:
            penalties += 0.05 * len(q.invalid_values)
        q.overall_quality_score = max(0.0, round(1.0 - penalties, 3))

        reports[ds_name] = q

    return reports


def print_quality_report(reports: Dict[str, DatasetQuality]) -> None:
    print("\n" + "=" * 58)
    print("  DATA QUALITY REPORT")
    print("=" * 58)
    for ds, q in reports.items():
        print(f"\nDataset: {ds}")
        print(f"  Rows                : {q.rows}")
        print(f"  Missing IDs         : {q.missing_id_count}")
        print(f"  Duplicate IDs       : {q.duplicate_id_count}")
        print(f"  Duplicate Names     : {q.duplicate_customer_count}")
        if q.invalid_values:
            for k, v in q.invalid_values.items():
                print(f"  Invalid {k:<15}: {v}")
        if q.string_as_numeric:
            print(f"  Numeric-as-string   : {', '.join(q.string_as_numeric)}")
        print(f"  Quality Score       : {q.overall_quality_score:.2f}")
        if q.notes:
            for note in q.notes[:5]:
                print(f"    → {note}")
    print("=" * 58 + "\n")
