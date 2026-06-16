#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge LightGBM predictions with recent news sentiment.

Example:
  python news_sentiment_overlay.py --prediction prediction_5d_lightgbm.csv --news news_recent.csv --output prediction_5d_with_news.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


POSITIVE_WORDS = [
    "利多", "看好", "上修", "調升", "買進", "增持", "優於預期", "創高", "新高",
    "大增", "成長", "轉強", "旺", "爆發", "接單", "滿載", "漲價", "擴產",
    "AI", "人工智慧", "先進製程", "高階", "復甦", "回溫", "獲利", "營收成長",
    "外資買超", "三大法人買超", "目標價調升", "配息", "得標", "訂單", "突破", "漲停",
]
NEGATIVE_WORDS = [
    "利空", "看壞", "下修", "調降", "賣出", "減持", "低於預期", "新低",
    "大減", "衰退", "轉弱", "淡季", "庫存", "砍單", "降價", "虧損",
    "營收下滑", "獲利下滑", "法說保守", "外資賣超", "三大法人賣超",
    "目標價調降", "處分", "裁罰", "罰款", "訴訟", "調查", "注意股",
    "警示股", "跌停", "違約", "停工", "停牌",
]
MAJOR_EVENT_WORDS = [
    "財報", "法說", "股東會", "除息", "除權", "配息", "併購", "增資",
    "減資", "處分", "裁罰", "注意股", "警示股", "營收", "EPS", "獲利", "虧損",
]


def count_hits(text: str, words: list[str]) -> int:
    text = "" if not isinstance(text, str) else text.lower()
    return sum(1 for w in words if w.lower() in text)


def classify_title(title: str) -> tuple[str, int, int, int, str]:
    pos = count_hits(title, POSITIVE_WORDS)
    neg = count_hits(title, NEGATIVE_WORDS)
    major = count_hits(title, MAJOR_EVENT_WORDS)
    raw = pos - neg
    if raw > 0:
        label = "偏多"
    elif raw < 0:
        label = "偏空"
    else:
        label = "中性"
    score = max(-100, min(100, raw * 25))
    return label, score, pos, neg, "是" if major > 0 else "否"


def summarize_news(news: pd.DataFrame) -> pd.DataFrame:
    empty_cols = [
        "stock_id", "新聞數量", "利多新聞數", "利空新聞數", "中性新聞數",
        "重大事件數", "新聞分數", "新聞面", "代表新聞", "新聞來源", "新聞時間", "新聞連結",
    ]
    if news.empty or "stock_id" not in news.columns:
        return pd.DataFrame(columns=empty_cols)

    news["stock_id"] = news["stock_id"].astype(str)
    if "news_title" not in news.columns:
        news["news_title"] = ""

    rows = []
    for stock_id, g in news.groupby("stock_id"):
        valid = g[g["news_title"].fillna("").astype(str).str.len() > 0].copy()
        if valid.empty:
            rows.append({
                "stock_id": stock_id, "新聞數量": 0, "利多新聞數": 0, "利空新聞數": 0,
                "中性新聞數": 0, "重大事件數": 0, "新聞分數": 0, "新聞面": "無新聞",
                "代表新聞": "", "新聞來源": "", "新聞時間": "", "新聞連結": "",
            })
            continue

        sentiments = valid["news_title"].apply(classify_title)
        valid["sentiment_label"] = sentiments.apply(lambda x: x[0])
        valid["sentiment_score"] = sentiments.apply(lambda x: x[1])
        valid["major_event"] = sentiments.apply(lambda x: x[4])
        valid["abs_score"] = valid["sentiment_score"].abs()

        news_count = len(valid)
        pos_count = int((valid["sentiment_label"] == "偏多").sum())
        neg_count = int((valid["sentiment_label"] == "偏空").sum())
        neutral_count = int((valid["sentiment_label"] == "中性").sum())
        major_count = int((valid["major_event"] == "是").sum())
        score = float(valid["sentiment_score"].mean())

        if score >= 15:
            label = "偏多"
        elif score <= -15:
            label = "偏空"
        else:
            label = "中性"

        rep = valid.sort_values("abs_score", ascending=False).iloc[0]
        rows.append({
            "stock_id": stock_id,
            "新聞數量": news_count,
            "利多新聞數": pos_count,
            "利空新聞數": neg_count,
            "中性新聞數": neutral_count,
            "重大事件數": major_count,
            "新聞分數": round(score, 2),
            "新聞面": label,
            "代表新聞": rep.get("news_title", ""),
            "新聞來源": rep.get("news_source", ""),
            "新聞時間": rep.get("news_time", ""),
            "新聞連結": rep.get("news_url", ""),
        })
    return pd.DataFrame(rows)


def pct_to_float(x) -> float:
    if pd.isna(x):
        return math.nan
    s = str(x).strip().replace("%", "")
    try:
        v = float(s)
        return v / 100.0 if v > 1 else v
    except Exception:
        return math.nan


def combined_judgement(prob_up: float, news_score: float, news_count: int) -> tuple[float, str, str]:
    tech_score = prob_up * 100 if not math.isnan(prob_up) else 50.0
    news_score_0_100 = 50 + news_score / 2
    news_weight = 0.0 if news_count <= 0 else 0.30
    combined = tech_score * (1 - news_weight) + news_score_0_100 * news_weight

    if combined >= 58:
        label = "偏多"
    elif combined <= 42:
        label = "偏空"
    else:
        label = "中性"

    if news_count <= 0:
        comment = "無近期新聞，主要依技術面判斷"
    elif tech_score >= 55 and news_score <= -25:
        comment = "技術面偏多，但新聞面有風險"
    elif tech_score <= 45 and news_score >= 25:
        comment = "技術面偏弱，但新聞面有支撐"
    elif label == "偏多":
        comment = "技術面與消息面綜合偏多"
    elif label == "偏空":
        comment = "技術面與消息面綜合偏空"
    else:
        comment = "技術面與消息面未形成明確方向"

    return round(combined, 2), label, comment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", default="prediction_5d_lightgbm.csv")
    ap.add_argument("--news", default="news_recent.csv")
    ap.add_argument("--output", default="prediction_5d_with_news.csv")
    args = ap.parse_args()

    pred_path = Path(args.prediction)
    if not pred_path.exists():
        raise FileNotFoundError(f"prediction file not found: {pred_path}")

    pred = pd.read_csv(pred_path, dtype={"stock_id": str})
    if "stock_id" not in pred.columns:
        raise ValueError("prediction CSV must contain stock_id")
    if "prob_up_5d" not in pred.columns:
        raise ValueError("prediction CSV must contain prob_up_5d")

    news = pd.read_csv(args.news, dtype={"stock_id": str}) if Path(args.news).exists() else pd.DataFrame()
    summary = summarize_news(news)

    out = pred.merge(summary, on="stock_id", how="left")
    defaults = {
        "新聞數量": 0, "利多新聞數": 0, "利空新聞數": 0, "中性新聞數": 0,
        "重大事件數": 0, "新聞分數": 0, "新聞面": "無新聞", "代表新聞": "",
        "新聞來源": "", "新聞時間": "", "新聞連結": "",
    }
    for c, v in defaults.items():
        out[c] = out[c].fillna(v) if c in out.columns else v

    combined = []
    for _, r in out.iterrows():
        combined.append(combined_judgement(
            pct_to_float(r.get("prob_up_5d")),
            float(r.get("新聞分數", 0) or 0),
            int(float(r.get("新聞數量", 0) or 0)),
        ))

    out["綜合分數"] = [x[0] for x in combined]
    out["綜合判斷"] = [x[1] for x in combined]
    out["綜合說明"] = [x[2] for x in combined]

    preferred = [
        "stock_id", "date", "close", "prediction_5d", "prob_up_5d", "prob_down_5d",
        "新聞面", "新聞分數", "新聞數量", "利多新聞數", "利空新聞數", "重大事件數",
        "綜合分數", "綜合判斷", "綜合說明", "代表新聞", "新聞來源", "新聞時間", "新聞連結",
    ]
    ordered = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[ordered]
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[news_overlay] saved {args.output}, rows={len(out)}")


if __name__ == "__main__":
    main()
