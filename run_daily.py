# -*- coding: utf-8 -*-
"""每日收盤後使用：增量更新資料 -> 訓練 LightGBM -> 轉 HTML。

這支是日常用，不是第一次完整回補用。

第一次完整回補完成後，下週一到五收盤後跑：
    python run_daily.py

比較快：
    python run_daily.py --sleep 0.5 --retries 3 --retry-sleep 5

如果只是測前 20 檔：
    python run_daily.py --max-stocks 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List


LOG_FILE = "daily_pipeline_log.txt"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def write_log(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def run_cmd(cmd: List[str], log_path: Path) -> None:
    header = "\n" + "=" * 90 + "\n"
    header += f"[{now()}] RUN: {quote_cmd(cmd)}\n"
    header += "=" * 90 + "\n"
    print(header, end="")
    write_log(log_path, header)

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
        write_log(log_path, line)

    code = p.wait()
    footer = f"\n[{now()}] EXIT CODE: {code}\n"
    print(footer, end="")
    write_log(log_path, footer)

    if code != 0:
        raise RuntimeError(f"命令失敗: {quote_cmd(cmd)}")


def check_files() -> None:
    required = [
        "update_prices_incremental.py",
        "train_lightgbm_5d.py",
        "csv_to_html.py",
        "local_data_loader.py",
        "stocks.txt",
    ]
    missing = [x for x in required if not Path(x).exists()]
    if missing:
        raise FileNotFoundError("缺少必要檔案:\n" + "\n".join(f"  - {x}" for x in missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily TWSTOCK incremental update pipeline.")
    parser.add_argument("--stock-file", default="stocks.txt")
    parser.add_argument("--csv-dir", default=".")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--start", default="2023-01-01", help="缺檔回補時的起始日")
    parser.add_argument("--end", default=dt.date.today().isoformat(), help="更新到哪一天，預設今天")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--backfill-missing", action="store_true", help="缺少個股 CSV 時才從 --start 補")
    parser.add_argument("--max-stocks", type=int, default=0, help="測試用，只處理前 N 檔")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=60)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(LOG_FILE)
    log_path.write_text(f"Daily pipeline started at {now()}\n", encoding="utf-8")

    check_files()

    print("TWSTOCK 每日增量流程")
    print(f"工作資料夾: {Path.cwd()}")
    print(f"日期上限: {args.end}")
    print(f"CSV 資料夾: {Path(args.csv_dir).resolve()}")
    print(f"HTML 入口: {(Path(args.output_dir) / 'html' / 'index.html').resolve()}")
    print()

    update_cmd = [
        sys.executable,
        "update_prices_incremental.py",
        "--stock-file",
        args.stock_file,
        "--csv-dir",
        args.csv_dir,
        "--start",
        args.start,
        "--end",
        args.end,
        "--sleep",
        str(args.sleep),
        "--retries",
        str(args.retries),
        "--retry-sleep",
        str(args.retry_sleep),
        "--combined",
    ]
    if args.backfill_missing:
        update_cmd.append("--backfill-missing")
    if args.max_stocks:
        update_cmd.extend(["--max-stocks", str(args.max_stocks)])

    try:
        run_cmd(update_cmd, log_path)

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                "train_lightgbm_5d.py",
                "--input",
                str(Path(args.csv_dir) / "all_price.csv"),
                "--output-dir",
                args.output_dir,
                "--top-n",
                str(args.top_n),
                "--min-history",
                str(args.min_history),
                "--num-boost-round",
                str(args.num_boost_round),
                "--early-stopping-rounds",
                str(args.early_stopping_rounds),
            ]
            run_cmd(train_cmd, log_path)

        if not args.skip_html:
            html_cmd = [
                sys.executable,
                "csv_to_html.py",
                "--csv-dir",
                args.output_dir,
                "--output-dir",
                str(Path(args.output_dir) / "html"),
                "--max-rows",
                "500",
                "--include",
                "prediction*.csv",
                "backtest*.csv",
                "feature_importance*.csv",
            ]
            run_cmd(html_cmd, log_path)

    except Exception as exc:
        print()
        print("每日流程失敗:", exc)
        print(f"詳細 log: {log_path.resolve()}")
        raise SystemExit(1)

    print()
    print("每日流程完成")
    print(f"詳細 log: {log_path.resolve()}")
    print(f"手機入口: {(Path(args.output_dir) / 'html' / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
