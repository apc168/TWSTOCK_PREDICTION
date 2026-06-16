#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove ETF *_price.csv files from TWSTOCK folder.

預設不直接刪除，而是移到 backup_etf_csv/，避免誤刪。
確認無誤後，可手動刪 backup_etf_csv。

用法：
    python remove_etf_csv.py

真的永久刪除：
    python remove_etf_csv.py --delete

指定資料夾：
    python remove_etf_csv.py --csv-dir "G:\\My Drive\\apc\\TWSTOCK"

也移除 ETF HTML：
    python remove_etf_csv.py --remove-html
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp"
MARKETS = {
    "twse": 2,
    "tpex": 4,
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\u3000", " ").strip()


def split_code_name(value: str) -> tuple[str, str] | None:
    value = normalize_text(value)
    match = re.match(r"^([0-9A-Z]{4,8})\s+(.+)$", value)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def is_etf_code(code: str) -> bool:
    return bool(re.fullmatch(r"\d{4,6}[A-Z]?", code))


def fetch_etf_ids(timeout: int = 30, sleep: float = 0.8) -> set[str]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    etf_ids: set[str] = set()

    for market, mode in MARKETS.items():
        print(f"Fetching ETF list: {market} strMode={mode} ...")
        response = session.get(ISIN_URL, params={"strMode": mode}, timeout=timeout)
        response.raise_for_status()
        response.encoding = "cp950"

        tables = pd.read_html(StringIO(response.text), header=0)
        if not tables:
            raise RuntimeError(f"No table found for market={market}")

        df = tables[0]
        code_name_col = df.columns[0]

        current_section = ""

        for _, row in df.iterrows():
            first = normalize_text(row.get(code_name_col, ""))

            parsed = split_code_name(first)
            if parsed is None:
                if first:
                    current_section = first
                continue

            code, _name = parsed

            if current_section == "ETF" and is_etf_code(code):
                etf_ids.add(code)

        if sleep > 0:
            time.sleep(sleep)

    return etf_ids


def remove_files(
    csv_dir: Path,
    etf_ids: set[str],
    delete: bool,
    backup_dir: Path,
    remove_html: bool,
) -> list[Path]:
    removed: list[Path] = []

    if not delete:
        backup_dir.mkdir(parents=True, exist_ok=True)

    for stock_id in sorted(etf_ids):
        candidates = [
            csv_dir / f"{stock_id}_price.csv",
        ]

        if remove_html:
            candidates.append(csv_dir / "html" / f"{stock_id}_price.html")

        for path in candidates:
            if not path.exists():
                continue

            if delete:
                path.unlink()
                removed.append(path)
                print(f"Deleted: {path}")
            else:
                # Preserve subfolder structure for html files.
                if path.parent.name == "html":
                    target_dir = backup_dir / "html"
                else:
                    target_dir = backup_dir
                target_dir.mkdir(parents=True, exist_ok=True)

                target = target_dir / path.name
                if target.exists():
                    target.unlink()

                shutil.move(str(path), str(target))
                removed.append(path)
                print(f"Moved: {path} -> {target}")

    return removed


def remove_etf_rows_from_all_price(csv_dir: Path, etf_ids: set[str], delete: bool) -> None:
    all_price = csv_dir / "all_price.csv"
    if not all_price.exists():
        return

    print()
    print("Checking all_price.csv ...")
    df = pd.read_csv(all_price, dtype={"stock_id": str})
    if "stock_id" not in df.columns:
        print("all_price.csv has no stock_id column; skipped.")
        return

    before = len(df)
    df["stock_id"] = df["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    filtered = df[~df["stock_id"].isin(etf_ids)].copy()
    removed_rows = before - len(filtered)

    if removed_rows == 0:
        print("No ETF rows found in all_price.csv.")
        return

    backup = csv_dir / f"all_price_before_remove_etf_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if not delete:
        shutil.copy2(all_price, backup)
        print(f"Backup all_price.csv: {backup}")

    filtered.to_csv(all_price, index=False, encoding="utf-8-sig")
    print(f"Updated all_price.csv: removed {removed_rows:,} ETF rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove ETF *_price.csv from TWSTOCK folder.")
    parser.add_argument("--csv-dir", default=".", help="TWSTOCK folder. Default: current folder.")
    parser.add_argument("--delete", action="store_true", help="Permanently delete instead of moving to backup_etf_csv.")
    parser.add_argument("--backup-dir", default="backup_etf_csv", help="Backup folder when not using --delete.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--remove-html", action="store_true", help="Also remove matching html/<stock_id>_price.html files.")
    parser.add_argument("--update-all-price", action="store_true", help="Also remove ETF rows from all_price.csv.")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    if not csv_dir.exists():
        raise FileNotFoundError(csv_dir)

    etf_ids = fetch_etf_ids(timeout=args.timeout)
    print()
    print(f"ETF ids found: {len(etf_ids)}")

    removed = remove_files(
        csv_dir=csv_dir,
        etf_ids=etf_ids,
        delete=args.delete,
        backup_dir=csv_dir / args.backup_dir,
        remove_html=args.remove_html,
    )

    if args.update_all_price:
        remove_etf_rows_from_all_price(csv_dir, etf_ids, delete=args.delete)

    print()
    print(f"Matched ETF CSV/HTML files removed or moved: {len(removed)}")
    if not args.delete:
        print(f"Backup folder: {(csv_dir / args.backup_dir).resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
