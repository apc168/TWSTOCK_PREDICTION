#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch recent Google News RSS headlines for TWSTOCK predictions.

Example:
  python news_fetcher.py --prediction prediction_5d_lightgbm.csv --output news_recent.csv --days 7 --top-n 300

For a watchlist:
  python news_fetcher.py --prediction prediction_5d_lightgbm.csv --output news_recent.csv --watchlist watchlist.txt --top-n 0
"""

from __future__ import annotations

import argparse
import html
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


USER_AGENT = "Mozilla/5.0 TWSTOCK-NewsFetcher/1.0"


def read_watchlist(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"watchlist not found: {p}")
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_targets(prediction_csv: str, watchlist: Optional[str], top_n: int) -> pd.DataFrame:
    df = pd.read_csv(prediction_csv, dtype={"stock_id": str})
    if "stock_id" not in df.columns:
        raise ValueError("prediction CSV must contain stock_id")

    watch_ids = read_watchlist(watchlist)
    if watch_ids:
        df = df[df["stock_id"].isin(watch_ids)].copy()

    if "prob_up_5d" in df.columns:
        df["_prob"] = pd.to_numeric(df["prob_up_5d"], errors="coerce")
        df = df.sort_values("_prob", ascending=False)

    if top_n and top_n > 0:
        df = df.head(top_n)

    keep = ["stock_id"]
    for c in ["stock_name", "name", "date", "close", "prob_up_5d"]:
        if c in df.columns:
            keep.append(c)
    return df[keep].drop_duplicates("stock_id")


def build_query(stock_id: str, stock_name: str, days: int) -> str:
    name_part = f' OR "{stock_name}"' if stock_name else ""
    return f'"{stock_id}"{name_part} 台股 股票 when:{days}d'


def rss_url(query: str) -> str:
    q = urllib.parse.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"


def parse_pubdate(pubdate: str) -> str:
    if not pubdate:
        return ""
    try:
        dt = datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return pubdate


def fetch_items(url: str, timeout: int) -> list[dict]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    channel = root.find("channel")
    if channel is None:
        return []

    rows = []
    for item in channel.findall("item"):
        title = html.unescape((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        pubdate = parse_pubdate((item.findtext("pubDate") or "").strip())
        source = ""
        source_node = item.find("source")
        if source_node is not None and source_node.text:
            source = html.unescape(source_node.text.strip())
        if title:
            rows.append({
                "news_title": title,
                "news_source": source,
                "news_time": pubdate,
                "news_url": link,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", default="prediction_5d_lightgbm.csv")
    ap.add_argument("--output", default="news_recent.csv")
    ap.add_argument("--watchlist", default=None)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--top-n", type=int, default=300, help="0 means all selected rows")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--max-news-per-stock", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args()

    targets = load_targets(args.prediction, args.watchlist, args.top_n)
    rows = []

    print(f"[news_fetcher] targets={len(targets)} output={args.output}")

    for idx, row in targets.reset_index(drop=True).iterrows():
        stock_id = str(row["stock_id"])
        stock_name = ""
        for c in ["stock_name", "name"]:
            if c in row and pd.notna(row[c]):
                stock_name = str(row[c]).strip()
                break

        query = build_query(stock_id, stock_name, args.days)
        try:
            items = fetch_items(rss_url(query), args.timeout)[: args.max_news_per_stock]
            for it in items:
                it.update({
                    "stock_id": stock_id,
                    "stock_name": stock_name,
                    "query": query,
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "error": "",
                })
                rows.append(it)
            print(f"[{idx+1}/{len(targets)}] {stock_id}: {len(items)}")
        except Exception as e:
            print(f"[{idx+1}/{len(targets)}] {stock_id}: ERROR {e}")
            rows.append({
                "stock_id": stock_id,
                "stock_name": stock_name,
                "news_title": "",
                "news_source": "",
                "news_time": "",
                "news_url": "",
                "query": query,
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
            })
        if args.sleep > 0:
            time.sleep(args.sleep)

    cols = ["stock_id", "stock_name", "news_title", "news_source", "news_time", "news_url", "query", "fetched_at", "error"]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.drop_duplicates(subset=["stock_id", "news_title", "news_url"], keep="first")
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[news_fetcher] saved {args.output}, rows={len(out)}")


if __name__ == "__main__":
    main()
