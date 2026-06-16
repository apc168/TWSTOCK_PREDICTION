#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate stocks.txt for Taiwan listed / OTC common stocks and ETFs.

Usage:
    python make_stocks_txt.py

Outputs:
    stocks.txt          # one stock/ETF id per line, for example.py --stock-file
    stocks_meta.csv     # metadata for review/debugging
"""

from __future__ import annotations

import argparse
import re
import time
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp"

MARKETS = {
    "twse": 2,  # listed securities
    "tpex": 4,  # OTC securities
}

INCLUDE_SECTIONS = {"股票", "ETF"}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\u3000", " ").strip()


def find_column(columns: Iterable[object], keywords: Iterable[str]) -> object | None:
    for col in columns:
        text = str(col)
        if any(keyword in text for keyword in keywords):
            return col
    return None


def split_code_name(value: str) -> tuple[str, str] | None:
    """
    ISIN table first column is usually:
        2330　台積電
        00631L 元大台灣50正2
    """
    value = normalize_text(value)
    match = re.match(r"^([0-9A-Z]{4,8})\s+(.+)$", value)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def is_common_stock_code(code: str) -> bool:
    # Keep normal listed/OTC common stocks. This intentionally excludes most
    # preferred shares / special securities with letters.
    return bool(re.fullmatch(r"\d{4}", code))


def is_etf_code(code: str) -> bool:
    # ETFs may include leveraged/inverse/commodity suffixes, e.g. 00631L, 00632R, 00635U.
    return bool(re.fullmatch(r"\d{4,6}[A-Z]?", code))


def fetch_market(session: requests.Session, market: str, mode: int, timeout: int) -> list[dict[str, str]]:
    response = session.get(ISIN_URL, params={"strMode": mode}, timeout=timeout)
    response.raise_for_status()
    response.encoding = "cp950"

    tables = pd.read_html(StringIO(response.text), header=0)
    if not tables:
        raise RuntimeError(f"No table found for market={market}")

    df = tables[0]
    code_name_col = df.columns[0]
    isin_col = find_column(df.columns, ["ISIN", "辨識"])
    listed_date_col = find_column(df.columns, ["上市日", "上櫃日", "登錄日"])
    market_col = find_column(df.columns, ["市場別"])
    industry_col = find_column(df.columns, ["產業別"])
    cfi_col = find_column(df.columns, ["CFI"])

    records: list[dict[str, str]] = []
    current_section = ""

    for _, row in df.iterrows():
        first = normalize_text(row.get(code_name_col, ""))

        # Section header rows look like "股票", "ETF", "認購權證", etc.
        split = split_code_name(first)
        if split is None:
            if first:
                current_section = first
            continue

        if current_section not in INCLUDE_SECTIONS:
            continue

        code, name = split

        if current_section == "股票" and not is_common_stock_code(code):
            continue

        if current_section == "ETF" and not is_etf_code(code):
            continue

        records.append(
            {
                "market": market,
                "section": current_section,
                "stock_id": code,
                "stock_name": name,
                "isin": normalize_text(row.get(isin_col, "")) if isin_col is not None else "",
                "listed_date": normalize_text(row.get(listed_date_col, "")) if listed_date_col is not None else "",
                "market_name": normalize_text(row.get(market_col, "")) if market_col is not None else "",
                "industry_category": normalize_text(row.get(industry_col, "")) if industry_col is not None else "",
                "cfi": normalize_text(row.get(cfi_col, "")) if cfi_col is not None else "",
            }
        )

    return records


def write_outputs(records: list[dict[str, str]], output_path: Path, meta_path: Path) -> None:
    if not records:
        raise RuntimeError("No stock/ETF ids were collected. Please check network access or TWSE ISIN page format.")

    df = pd.DataFrame(records)

    # De-duplicate while preserving official market order: TWSE first, then TPEx.
    df = df.drop_duplicates(subset=["stock_id"], keep="first").reset_index(drop=True)

    output_path.write_text("\n".join(df["stock_id"].astype(str).tolist()) + "\n", encoding="utf-8")
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    counts = df.groupby(["market", "section"]).size().reset_index(name="count")
    print("Generated:", output_path)
    print("Metadata:", meta_path)
    print()
    print(counts.to_string(index=False))
    print()
    print(f"Total ids: {len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TWSE/TPEx stocks.txt including ETFs.")
    parser.add_argument("--output", default="stocks.txt", help="Output stock id list. Default: stocks.txt")
    parser.add_argument("--meta", default="stocks_meta.csv", help="Metadata CSV. Default: stocks_meta.csv")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout seconds. Default: 30")
    parser.add_argument("--sleep", type=float, default=0.8, help="Sleep seconds between TWSE and TPEx requests.")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_records: list[dict[str, str]] = []

    for market, mode in MARKETS.items():
        print(f"Fetching {market} strMode={mode} ...")
        all_records.extend(fetch_market(session, market=market, mode=mode, timeout=args.timeout))
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_outputs(
        records=all_records,
        output_path=Path(args.output),
        meta_path=Path(args.meta),
    )


if __name__ == "__main__":
    main()
