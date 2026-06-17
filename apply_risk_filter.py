# -*- coding: utf-8 -*-
"""Apply manual news / material-event risk guardrails to model predictions.

This script takes the model output `prediction_5d_lightgbm.csv` and merges it
with a manually maintained risk watchlist. It creates `final_stock_radar.csv`,
which should be used as the final report for human review.

Why this exists:
- The LightGBM model only understands price / volume / technical indicators.
- It does not know about delisting risk, trading suspension, litigation,
  accounting issues, or other material events.
- This guardrail lets you block or downgrade risky stocks after the model ranks
  them.

Expected watchlist columns:
    stock_id,risk_level,reason,source,event_date,expire_date

Only `stock_id` is required. Missing values are filled safely.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
VALID_RISK_LEVELS = set(RISK_ORDER)


def normalize_stock_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def normalize_risk_level(value: object) -> str:
    if pd.isna(value):
        return "LOW"
    text = str(value).strip().upper()
    mapping = {
        "H": "HIGH",
        "HIGH_RISK": "HIGH",
        "高": "HIGH",
        "高風險": "HIGH",
        "M": "MEDIUM",
        "MID": "MEDIUM",
        "中": "MEDIUM",
        "中風險": "MEDIUM",
        "L": "LOW",
        "低": "LOW",
        "低風險": "LOW",
    }
    text = mapping.get(text, text)
    return text if text in VALID_RISK_LEVELS else "LOW"


def parse_date(value: object) -> Optional[dt.date]:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def load_watchlist(path: Path, today: dt.date) -> pd.DataFrame:
    columns = [
        "stock_id",
        "news_risk_level",
        "news_risk_reason",
        "news_source",
        "news_event_date",
        "risk_expire_date",
    ]

    if not path.exists():
        return pd.DataFrame(columns=columns)

    risk = pd.read_csv(path, dtype=str).fillna("")
    if risk.empty or "stock_id" not in risk.columns:
        return pd.DataFrame(columns=columns)

    risk = risk.copy()
    risk["stock_id"] = risk["stock_id"].map(normalize_stock_id)

    # Accept both English and output-style column names.
    if "risk_level" in risk.columns:
        risk["news_risk_level"] = risk["risk_level"].map(normalize_risk_level)
    elif "news_risk_level" in risk.columns:
        risk["news_risk_level"] = risk["news_risk_level"].map(normalize_risk_level)
    else:
        risk["news_risk_level"] = "LOW"

    if "reason" in risk.columns:
        risk["news_risk_reason"] = risk["reason"]
    elif "news_risk_reason" not in risk.columns:
        risk["news_risk_reason"] = ""

    if "source" in risk.columns:
        risk["news_source"] = risk["source"]
    elif "news_source" not in risk.columns:
        risk["news_source"] = "manual"

    if "event_date" in risk.columns:
        risk["news_event_date"] = risk["event_date"]
    elif "news_event_date" not in risk.columns:
        risk["news_event_date"] = ""

    if "expire_date" in risk.columns:
        risk["risk_expire_date"] = risk["expire_date"]
    elif "risk_expire_date" not in risk.columns:
        risk["risk_expire_date"] = ""

    # Ignore expired rows.
    keep_rows: List[bool] = []
    for _, row in risk.iterrows():
        expire_date = parse_date(row.get("risk_expire_date", ""))
        keep_rows.append(expire_date is None or expire_date >= today)
    risk = risk.loc[keep_rows].copy()

    if risk.empty:
        return pd.DataFrame(columns=columns)

    # If one stock has multiple risk rows, keep the highest risk.
    risk["_risk_score"] = risk["news_risk_level"].map(RISK_ORDER).fillna(0).astype(int)
    risk = risk.sort_values(["stock_id", "_risk_score"], ascending=[True, False])
    risk = risk.drop_duplicates("stock_id", keep="first")

    return risk[columns]


def choose_prob_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["prob_up_5d", "5日看漲機率", "prob_up", "score"]:
        if col in df.columns:
            return col
    return None


def apply_risk_filter(prediction: pd.DataFrame, watchlist: pd.DataFrame) -> pd.DataFrame:
    pred = prediction.copy()
    if "stock_id" not in pred.columns:
        raise ValueError("prediction file must contain column: stock_id")

    pred["stock_id"] = pred["stock_id"].map(normalize_stock_id)

    if watchlist.empty:
        pred["news_risk_level"] = "LOW"
        pred["news_risk_reason"] = ""
        pred["news_event_date"] = ""
        pred["news_source"] = ""
        pred["risk_expire_date"] = ""
    else:
        pred = pred.merge(watchlist, on="stock_id", how="left")
        pred["news_risk_level"] = pred["news_risk_level"].fillna("LOW")
        pred["news_risk_reason"] = pred["news_risk_reason"].fillna("")
        pred["news_event_date"] = pred["news_event_date"].fillna("")
        pred["news_source"] = pred["news_source"].fillna("")
        pred["risk_expire_date"] = pred["risk_expire_date"].fillna("")

    pred["news_risk_level"] = pred["news_risk_level"].map(normalize_risk_level)
    pred["risk_score"] = pred["news_risk_level"].map(RISK_ORDER).fillna(0).astype(int)

    def signal(level: str) -> str:
        if level == "HIGH":
            return "BLOCKED"
        if level == "MEDIUM":
            return "WATCH_WITH_CAUTION"
        return "WATCH"

    pred["final_signal"] = pred["news_risk_level"].map(signal)
    pred["is_blocked"] = pred["final_signal"].eq("BLOCKED")

    prob_col = choose_prob_column(pred)
    if prob_col:
        pred["_rank_score"] = pd.to_numeric(pred[prob_col], errors="coerce").fillna(0.0)
    else:
        pred["_rank_score"] = 0.0

    # Sort actionable rows first, then riskier rows, then higher model probability.
    pred = pred.sort_values(
        ["is_blocked", "risk_score", "_rank_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    # Use nullable integer dtype. Do not initialize with an empty string,
    # otherwise newer pandas string dtype will reject integer rank values.
    pred["final_rank"] = pd.Series([pd.NA] * len(pred), dtype="Int64")
    actionable = ~pred["is_blocked"]
    if actionable.any():
        pred.loc[actionable, "final_rank"] = list(range(1, int(actionable.sum()) + 1))

    # Put guardrail columns near the front.
    front = [
        "final_rank",
        "final_signal",
        "news_risk_level",
        "news_risk_reason",
        "news_event_date",
        "news_source",
        "risk_expire_date",
        "is_blocked",
    ]
    remaining = [c for c in pred.columns if c not in front and not c.startswith("_")]
    return pred[front + remaining]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply risk watchlist to prediction output.")
    parser.add_argument("--prediction", default="prediction_5d_lightgbm.csv")
    parser.add_argument("--risk-watchlist", default="risk_watchlist.csv")
    parser.add_argument("--output", default="final_stock_radar.csv")
    parser.add_argument("--blocked-output", default="blocked_stock_radar.csv")
    parser.add_argument("--today", default=dt.date.today().isoformat())
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_path = Path(args.prediction)
    risk_path = Path(args.risk_watchlist)
    output_path = Path(args.output)
    blocked_output_path = Path(args.blocked_output)
    today = pd.to_datetime(args.today).date()

    if not prediction_path.exists():
        raise FileNotFoundError(f"prediction file not found: {prediction_path}")

    prediction = pd.read_csv(prediction_path, dtype={"stock_id": str})
    watchlist = load_watchlist(risk_path, today=today)
    final = apply_risk_filter(prediction, watchlist)

    final.to_csv(output_path, index=False, encoding="utf-8-sig")

    blocked = final[final["final_signal"].eq("BLOCKED")].copy()
    blocked.to_csv(blocked_output_path, index=False, encoding="utf-8-sig")

    summary = final["final_signal"].value_counts().to_dict()
    print(f"Saved final radar: {output_path} ({len(final):,} rows)")
    print(f"Saved blocked list: {blocked_output_path} ({len(blocked):,} rows)")
    print("Signal summary:", summary)


if __name__ == "__main__":
    main()
