"""
aggregation_engine.py
═══════════════════════════════════════════════════════════════════════════════
Generic, schema-aware aggregation framework.

Supported operations:
  COUNT, COUNT_DISTINCT, SUM, AVERAGE, MEDIAN, MIN, MAX, PERCENTAGE, RATIO,
  STD, VARIANCE

A FieldResolver maps natural-language phrases to actual DataFrame columns
so adding a new CSV does not require source-code changes.

ALL arithmetic is done with pandas — the scratch LLM never touches numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────── field resolver ──────────────────────────

@dataclass
class FieldSpec:
    dataset: str
    column: str
    dtype: str    # "numeric" | "text" | "date"


# Column aliases: maps (normalised alias) -> FieldSpec
_ALIAS_MAP: Dict[str, FieldSpec] = {}


def _norm_alias(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def build_alias_map(dfs: Dict[str, pd.DataFrame]) -> None:
    """Called once at startup. Builds the alias map from actual CSV columns."""
    global _ALIAS_MAP
    _ALIAS_MAP = {}

    _builtin_aliases = {
        # Feedback
        "rating":           ("Feedback", "Rating", "numeric"),
        "feedback score":   ("Feedback", "Rating", "numeric"),
        "score":            ("Feedback", "Rating", "numeric"),
        "review score":     ("Feedback", "Rating", "numeric"),
        # Appointment
        "appointment status": ("Appointment", "Status", "text"),
        "booking status":     ("Appointment", "Status", "text"),
        # Enquiry
        "enquiry status":   ("Enquiry", "Status", "text"),
        "city":             ("Enquiry", "City / State", "text"),
        "state":            ("Enquiry", "City / State", "text"),
        "location":         ("Enquiry", "City / State", "text"),
        "vehicle":          ("Enquiry", "Vehicle Name / Model", "text"),
        "model":            ("Enquiry", "Vehicle Name / Model", "text"),
        "payment":          ("Enquiry", "Payment Mode", "text"),
        "payment mode":     ("Enquiry", "Payment Mode", "text"),
    }

    for alias, (ds, col, dtype) in _builtin_aliases.items():
        if ds in dfs and col in dfs[ds].columns:
            _ALIAS_MAP[_norm_alias(alias)] = FieldSpec(dataset=ds, column=col, dtype=dtype)

    # Auto-discover numeric columns from actual data
    for ds, df in dfs.items():
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                key = _norm_alias(col)
                if key not in _ALIAS_MAP:
                    _ALIAS_MAP[key] = FieldSpec(dataset=ds, column=col, dtype="numeric")
            # text columns → add column name as alias too
            key = _norm_alias(col)
            if key not in _ALIAS_MAP:
                dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text"
                _ALIAS_MAP[key] = FieldSpec(dataset=ds, column=col, dtype=dtype)


def resolve_field(phrase: str) -> Optional[FieldSpec]:
    """Find the best FieldSpec for a natural-language phrase."""
    p = _norm_alias(phrase)
    if p in _ALIAS_MAP:
        return _ALIAS_MAP[p]
    # Longest-suffix match
    best, best_len = None, 0
    for alias, spec in _ALIAS_MAP.items():
        if alias in p or p in alias:
            if len(alias) > best_len:
                best, best_len = spec, len(alias)
    return best


# ─────────────────────────────────── aggregation result ──────────────────────

@dataclass
class AggregationResult:
    operation: str
    dataset: str
    column: Optional[str]
    value: object          # the actual computed result (number / dict / list)
    facts: str             # human-readable fact string (no hallucination possible)
    row_count: int
    confidence: float      # 0–1


# ─────────────────────────────────── operations ──────────────────────────────

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def compute_aggregation(
    operation: str,
    df: pd.DataFrame,
    column: Optional[str],
    dataset: str,
    filters: Optional[pd.Series] = None,
    percentile: float = 0.5,
    top_k: int = 5,
) -> AggregationResult:
    """
    Execute one aggregation operation deterministically with pandas.

    Parameters
    ----------
    operation : COUNT | COUNT_DISTINCT | SUM | AVERAGE | MEDIAN |
                MIN | MAX | PERCENTAGE | RATIO | STD | VARIANCE
    df        : the DataFrame to aggregate (already filtered if needed)
    column    : the specific column (None for COUNT on whole df)
    dataset   : name for logging
    filters   : optional boolean mask for an additional row filter
    percentile: used for MEDIAN (0.5) or arbitrary PERCENTILE
    """
    if filters is not None:
        df = df[filters]

    n = len(df)

    def _series():
        if column and column in df.columns:
            return _to_numeric(df[column]) if operation not in ("COUNT", "COUNT_DISTINCT") else df[column].dropna()
        return pd.Series([], dtype=float)

    col_label = column or "rows"
    op = operation.upper()

    if op == "COUNT":
        return AggregationResult(
            operation=op, dataset=dataset, column=column,
            value=n, facts=f"Count of {dataset}: {n} record(s).",
            row_count=n, confidence=1.0,
        )

    if op == "COUNT_DISTINCT":
        s = _series()
        val = int(s.nunique())
        return AggregationResult(
            operation=op, dataset=dataset, column=column,
            value=val, facts=f"Distinct values in {col_label} ({dataset}): {val}.",
            row_count=n, confidence=1.0,
        )

    s = _to_numeric(df[column]) if column and column in df.columns else pd.Series([], dtype=float)

    if s.empty:
        return AggregationResult(
            operation=op, dataset=dataset, column=column,
            value=None, facts=f"No numeric data found in column '{col_label}' of {dataset}.",
            row_count=n, confidence=0.1,
        )

    if op == "SUM":
        val = round(float(s.sum()), 4)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Sum of {col_label} ({dataset}): {val} (from {len(s)} rows).",
            row_count=n, confidence=1.0,
        )
    if op == "AVERAGE":
        val = round(float(s.mean()), 4)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Average {col_label} ({dataset}): {val} (from {len(s)} rows).",
            row_count=n, confidence=1.0,
        )
    if op == "MEDIAN":
        val = round(float(s.median()), 4)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Median {col_label} ({dataset}): {val} (from {len(s)} rows).",
            row_count=n, confidence=1.0,
        )
    if op == "MIN":
        val = round(float(s.min()), 4)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Minimum {col_label} ({dataset}): {val}.",
            row_count=n, confidence=1.0,
        )
    if op == "MAX":
        val = round(float(s.max()), 4)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Maximum {col_label} ({dataset}): {val}.",
            row_count=n, confidence=1.0,
        )
    if op == "STD":
        val = round(float(s.std()), 4)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Standard deviation of {col_label} ({dataset}): {val}.",
            row_count=n, confidence=1.0,
        )
    if op == "VARIANCE":
        val = round(float(s.var()), 4)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Variance of {col_label} ({dataset}): {val}.",
            row_count=n, confidence=1.0,
        )
    if op == "PERCENTAGE":
        # percentage of filtered rows to total rows in original df
        val = round(100.0 * n / len(df) if len(df) else 0.0, 2)
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Percentage matching filter ({dataset}): {val}% ({n} of {len(df)} rows).",
            row_count=n, confidence=0.9,
        )
    if op == "RATIO":
        # ratio of two subsets — caller provides filtered df A and df B
        # for now just compute ratio of filtered/total
        val = round(n / len(df), 4) if len(df) else 0.0
        return AggregationResult(
            operation=op, dataset=dataset, column=column, value=val,
            facts=f"Ratio ({dataset}): {val} ({n} / {len(df)}).",
            row_count=n, confidence=0.8,
        )

    return AggregationResult(
        operation=op, dataset=dataset, column=column,
        value=None, facts=f"Unsupported aggregation operation: {op}.",
        row_count=n, confidence=0.0,
    )
