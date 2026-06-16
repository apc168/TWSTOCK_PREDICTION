# -*- coding: utf-8 -*-
"""增量更新台股 CSV：只補每檔 CSV 最後日期之後的新資料。

用途：
    第一次完整回補完成後，之後每天收盤後只跑這支，不要再全量重抓。

基本用法：
    python update_prices_incremental.py --combined

若有缺少的個股 CSV，預設會跳過，避免不小心又跑完整歷史回補。
如果真的要補缺檔：
    python update_prices_incremental.py --combined --backfill-missing

建議搭配：
    python run_daily.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from local_data_loader import DataLoader

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def split_stock_ids(raw: str) -> List[str]:
    ids: List[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        sid = chunk.strip()
        if sid:
            ids.append(sid)
    return ids


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_stock_ids(stock_file: str, max_stocks: int = 0) -> List[str]:
    path = Path(stock_file)
    if not path.exists():
        raise FileNotFoundError(f"找不到股票清單: {path}")

    ids: List[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.extend(split_stock_ids(line))

    ids = dedupe_keep_order(ids)
    if max_stocks and max_stocks > 0:
        ids = ids[:max_stocks]
    return ids


def read_price_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype={"stock_id": str})
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df.dropna(subset=["date"])
    except Exception:
        return None


def add_moving_average(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close" not in df.columns:
        return df
    df = df.copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.sort_values("date")
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    return df


def next_day(date_str: str) -> str:
    return (dt.date.fromisoformat(date_str) + dt.timedelta(days=1)).isoformat()


def update_one_stock(
    dl: DataLoader,
    stock_id: str,
    csv_dir: Path,
    start: str,
    end: str,
    backfill_missing: bool,
) -> Tuple[str, str, int]:
    """Return: status, message, new_rows."""
    path = csv_dir / f"{stock_id}_price.csv"
    existing = read_price_csv(path)

    if existing is None or existing.empty:
        if not backfill_missing:
            return "missing_skipped", "CSV 不存在，預設跳過；若要補缺檔請加 --backfill-missing", 0
        fetch_start = start
        existing = pd.DataFrame()
    else:
        last_date = str(existing["date"].max())
        fetch_start = next_day(last_date)
        if fetch_start > end:
            return "up_to_date", f"已是最新: {last_date}", 0

    new_df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=fetch_start, end_date=end)
    if new_df.empty:
        return "no_new_data", f"{fetch_start} ~ {end} 查無新資料", 0

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    combined["stock_id"] = combined["stock_id"].astype(str)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.dropna(subset=["date"])
    combined = combined.drop_duplicates(subset=["stock_id", "date"], keep="last")
    combined = combined.sort_values(["stock_id", "date"]).reset_index(drop=True)
    combined = add_moving_average(combined)

    csv_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")

    return "updated", f"補 {fetch_start} ~ {end}", len(new_df)


def rebuild_all_price(stock_ids: List[str], csv_dir: Path, output: str = "all_price.csv") -> Path:
    frames = []
    for sid in stock_ids:
        path = csv_dir / f"{sid}_price.csv"
        df = read_price_csv(path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("沒有可合併的 *_price.csv")

    combined = pd.concat(frames, ignore_index=True)
    if "stock_id" in combined.columns:
        combined["stock_id"] = combined["stock_id"].astype(str)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    sort_cols = [c for c in ["stock_id", "date"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols)

    output_path = csv_dir / output
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只補每檔股票 CSV 最後日期之後的新資料")
    parser.add_argument("--stock-file", default="stocks.txt")
    parser.add_argument("--csv-dir", default=".")
    parser.add_argument("--start", default="2023-01-01", help="缺檔回補時的起始日，預設 2023-01-01")
    parser.add_argument("--end", default=dt.date.today().isoformat(), help="更新到哪一天，預設今天")
    parser.add_argument("--sleep", type=float, default=0.5, help="request 間隔秒數，日常更新可用 0.5~1")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--backfill-missing", action="store_true", help="遇到缺少的 CSV 時，從 --start 回補")
    parser.add_argument("--combined", action="store_true", help="更新後重建 all_price.csv")
    parser.add_argument("--max-stocks", type=int, default=0, help="只處理前 N 檔，測試用")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    csv_dir = Path(args.csv_dir)
    stock_ids = load_stock_ids(args.stock_file, args.max_stocks)
    dl = DataLoader(sleep=args.sleep, retries=args.retries, retry_sleep=args.retry_sleep)

    print("增量更新股票 CSV")
    print(f"股票數量: {len(stock_ids)}")
    print(f"日期上限: {args.end}")
    print(f"CSV 資料夾: {csv_dir.resolve()}")
    print(f"缺檔處理: {'回補缺檔' if args.backfill_missing else '跳過缺檔'}")
    print()

    rows = []
    counts = {}

    for i, stock_id in enumerate(stock_ids, start=1):
        print(f"[{i}/{len(stock_ids)}] 檢查 {stock_id} ...")
        try:
            status, message, new_rows = update_one_stock(
                dl=dl,
                stock_id=stock_id,
                csv_dir=csv_dir,
                start=args.start,
                end=args.end,
                backfill_missing=args.backfill_missing,
            )
        except Exception as exc:
            status, message, new_rows = "failed", str(exc), 0

        counts[status] = counts.get(status, 0) + 1
        rows.append({"stock_id": stock_id, "status": status, "message": message, "new_rows": new_rows})

        if status == "updated":
            print(f"  已更新: {message}，新增 {new_rows} 筆")
        elif status == "failed":
            print(f"  失敗: {message}")
        else:
            print(f"  {message}")

    report = pd.DataFrame(rows)
    report_path = csv_dir / "daily_update_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print()
    print("更新摘要:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"明細: {report_path}")

    if args.combined:
        print()
        print("重建 all_price.csv ...")
        combined_path = rebuild_all_price(stock_ids, csv_dir)
        print(f"已輸出: {combined_path}")

    if counts.get("failed", 0):
        print()
        print("有失敗項目，可之後再跑同一個指令；已更新成功的檔案會保留。")


if __name__ == "__main__":
    main()
