# -*- coding: utf-8 -*-
"""local_data_loader 批次範例 — 支援多檔台股 CSV、重試、續跑。

基本用法：
    python example.py

指定多檔：
    python example.py --stock-ids 2330,2317,009816

用文字檔指定股票清單（一行一個股票代號）：
    python example.py --stock-file stocks.txt

大量批次建議：
    python example.py --stock-file stocks.txt --sleep 3 --retries 8 --retry-sleep 20

重新抓已存在的 CSV：
    python example.py --stock-file stocks.txt --force
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from local_data_loader import DataLoader

try:
    sys.stdout.reconfigure(encoding="utf-8")  # 讓中文在終端機正常顯示
except AttributeError:
    pass


# ── 預設參數：直接 python example.py 時會使用這裡 ───────────────
DEFAULT_STOCK_IDS = ["009816", "2330"]
DEFAULT_START = "2026-05-01"
DEFAULT_END = "2026-06-11"
DEFAULT_OUT_DIR = "."


def _split_stock_ids(raw: str) -> List[str]:
    """把 '2330, 2317\n009816' 轉成 ['2330', '2317', '009816']。"""
    ids: List[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        stock_id = chunk.strip()
        if stock_id:
            ids.append(stock_id)
    return ids


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_stock_ids(stock_ids: Optional[str], stock_file: Optional[str]) -> List[str]:
    """從 --stock-ids、--stock-file 或預設清單取得股票代號。"""
    ids: List[str] = []

    if stock_ids:
        ids.extend(_split_stock_ids(stock_ids))

    if stock_file:
        path = Path(stock_file)
        if not path.exists():
            raise FileNotFoundError(f"找不到股票清單檔案: {path}")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 支援一行一個，也支援一行多個逗號分隔
            ids.extend(_split_stock_ids(line))

    if not ids:
        ids = DEFAULT_STOCK_IDS[:]

    return _dedupe_keep_order(ids)


def add_moving_average(price: pd.DataFrame) -> pd.DataFrame:
    """保留原本範例的 ma5 / ma20 分析欄位。"""
    if price.empty or "close" not in price.columns:
        return price

    price = price.copy()
    price["ma5"] = price["close"].rolling(5).mean()
    price["ma20"] = price["close"].rolling(20).mean()
    return price


def save_one_stock(
    dl: DataLoader,
    stock_id: str,
    start: str,
    end: str,
    out_dir: Path,
) -> Tuple[Optional[pd.DataFrame], Optional[Path], Optional[str]]:
    """抓一檔股票、加均線、輸出 CSV。"""
    try:
        price = dl.taiwan_stock_daily(stock_id, start, end)
        if price.empty:
            return None, None, "查無資料"

        price = add_moving_average(price)

        # 確保 Excel 開啟中文/欄位不亂碼
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{stock_id}_price.csv"
        price.to_csv(output_path, index=False, encoding="utf-8-sig")

        return price, output_path, None
    except Exception as exc:  # 批次時單檔失敗不應中斷全部
        return None, None, str(exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批次產生多檔台股 CSV（使用公開市場資料端點）")
    parser.add_argument(
        "--stock-ids",
        help="逗號分隔股票代號，例如: 2330,2317,009816。未填則使用 DEFAULT_STOCK_IDS。",
    )
    parser.add_argument(
        "--stock-file",
        help="股票代號文字檔，一行一個；空行與 # 開頭註解會略過。",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="開始日期，例如: 2026-05-01")
    parser.add_argument("--end", default=DEFAULT_END, help="結束日期，例如: 2026-06-11")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="輸出資料夾，預設為目前資料夾。",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="額外輸出 all_price.csv，將所有股票合併在同一個 CSV。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="每次 request 間隔秒數；大量批次建議 2~5 秒。預設 1.5。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="單次 request 失敗時重試次數。預設 5。",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="第一次重試前等待秒數，後續會自動加長。預設 5。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使 {stock_id}_price.csv 已存在，也重新抓取。",
    )
    parser.add_argument(
        "--show-tail",
        action="store_true",
        help="每檔成功後顯示最後 5 筆資料；大量批次時建議不要開。",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=0,
        help="只抓前 N 檔，方便測試。0 表示不限制。",
    )
    return parser


def read_existing_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, dtype={"stock_id": str})
    except Exception:
        return None


def main() -> None:
    args = build_parser().parse_args()

    stock_ids = load_stock_ids(args.stock_ids, args.stock_file)
    if args.max_stocks and args.max_stocks > 0:
        stock_ids = stock_ids[: args.max_stocks]

    out_dir = Path(args.out_dir)
    dl = DataLoader(sleep=args.sleep, retries=args.retries, retry_sleep=args.retry_sleep)

    print("批次產生台股 CSV")
    print(f"股票數量: {len(stock_ids)}")
    print(f"日期區間: {args.start} ~ {args.end}")
    print(f"輸出資料夾: {out_dir.resolve()}")
    print(f"request 間隔: {args.sleep}s；重試: {args.retries} 次；retry-sleep: {args.retry_sleep}s")
    if not args.force:
        print("續跑模式: 已存在的 *_price.csv 會跳過；若要重抓請加 --force")
    print()

    all_prices: List[pd.DataFrame] = []
    failed: List[Tuple[str, str]] = []
    skipped = 0

    for index, stock_id in enumerate(stock_ids, start=1):
        output_path = out_dir / f"{stock_id}_price.csv"
        print(f"[{index}/{len(stock_ids)}] 抓取 {stock_id} ...")

        if output_path.exists() and not args.force:
            skipped += 1
            print(f"  已存在，跳過: {output_path}")
            if args.combined:
                existing = read_existing_csv(output_path)
                if existing is not None and not existing.empty:
                    all_prices.append(existing)
            continue

        price, output_path, error = save_one_stock(dl, stock_id, args.start, args.end, out_dir)

        if error:
            failed.append((stock_id, error))
            print(f"  失敗: {error}")
            continue

        assert price is not None and output_path is not None
        all_prices.append(price)

        if args.show_tail:
            print(price.tail())
            if {"date", "close", "ma5", "ma20"}.issubset(price.columns):
                print(price[["date", "close", "ma5", "ma20"]].tail())

        print(f"  已存檔: {output_path}")

    if args.combined and all_prices:
        combined = pd.concat(all_prices, ignore_index=True)
        sort_cols = [c for c in ["stock_id", "date"] if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols)
        combined_path = out_dir / "all_price.csv"
        combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"合併檔已存檔: {combined_path}")

    if failed:
        failed_path = out_dir / "failed_price.csv"
        pd.DataFrame(failed, columns=["stock_id", "reason"]).to_csv(
            failed_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"失敗清單已存檔: {failed_path}")

    success_count = len(stock_ids) - len(failed) - skipped
    print(f"\n完成: 新抓成功 {success_count} 檔，跳過 {skipped} 檔，失敗 {len(failed)} 檔")

    if failed:
        print("\n如果失敗原因仍是 empty/non-JSON response，通常是資料源暫時限流。")
        print("建議續跑指令：python example.py --stock-file stocks.txt --sleep 3 --retries 8 --retry-sleep 20")


if __name__ == "__main__":
    main()
