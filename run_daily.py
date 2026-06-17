# -*- coding: utf-8 -*-
"""每日收盤後使用：增量更新資料 -> 訓練 LightGBM -> 轉 HTML。

這支是日常用，不是第一次完整回補用。

重點：
- 預設只補每檔股票 CSV 最後日期之後的新資料。
- 預設不會從 2023-01-01 重新抓歷史資料。
- 如果 prices/ 裡沒有既有歷史 CSV，會直接停止，避免誤跑完整回補。
- 只有明確加上 --backfill-missing 時，才允許缺檔個股從 --start 回補。
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
        "apply_risk_filter.py",
        "csv_to_html.py",
        "stocks.txt",
    ]

    missing = [x for x in required if not Path(x).exists()]

    if not Path("local_data_loader.py").exists():
        missing.append("local_data_loader.py")

    if missing:
        raise FileNotFoundError("缺少必要檔案:\n" + "\n".join(f"  - {x}" for x in missing))


def ensure_daily_mode_is_safe(csv_dir: Path, backfill_missing: bool) -> None:
    """避免日常更新誤變成完整歷史回補。"""
    price_files = [
        p for p in csv_dir.glob("*_price.csv")
        if p.name != "all_price.csv"
    ]

    if price_files:
        return

    if backfill_missing:
        print("警告：目前找不到既有 *_price.csv，但你有加 --backfill-missing。")
        print("這會讓缺檔股票從 --start 開始回補歷史資料。")
        return

    raise RuntimeError(
        "\n".join(
            [
                f"在 {csv_dir.resolve()} 找不到任何既有 *_price.csv。",
                "為了避免每日更新誤跑完整歷史回補，流程已停止。",
                "",
                "如果你已經抓完歷史資料，請確認：",
                "  1. 你是在原本有 prices/ 資料的資料夾執行",
                "  2. 或確認 --csv-dir 指到正確的歷史資料資料夾",
                "",
                "日常更新建議：",
                "  python run_daily.py --csv-dir prices --output-dir .",
                "",
                "如果你真的要重新回補缺檔，才使用：",
                "  python run_daily.py --csv-dir prices --output-dir . --backfill-missing",
            ]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily TWSTOCK incremental update pipeline.")
    parser.add_argument("--stock-file", default="stocks.txt")
    parser.add_argument("--csv-dir", default=".")
    parser.add_argument("--output-dir", default=".")

    # start 只在 --backfill-missing 時使用；日常更新不會傳給 update_prices_incremental.py。
    parser.add_argument("--start", default="2023-01-01", help="只有缺檔回補時才使用的起始日")
    parser.add_argument("--end", default=dt.date.today().isoformat(), help="更新到哪一天，預設今天")

    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)

    parser.add_argument(
        "--backfill-missing",
        action="store_true",
        help="只有明確指定時，才允許缺少個股 CSV 時從 --start 回補",
    )

    parser.add_argument("--max-stocks", type=int, default=0, help="測試用，只處理前 N 檔")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-history", type=int, default=60)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-risk-filter", action="store_true")
    parser.add_argument("--risk-watchlist", default="risk_watchlist.csv")
    parser.add_argument("--final-output", default="final_stock_radar.csv")
    parser.add_argument("--skip-html", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(LOG_FILE)
    log_path.write_text(f"Daily pipeline started at {now()}\n", encoding="utf-8")

    check_files()

    csv_dir = Path(args.csv_dir)
    ensure_daily_mode_is_safe(csv_dir, args.backfill_missing)

    print("TWSTOCK 每日增量流程")
    print(f"工作資料夾: {Path.cwd()}")
    print(f"日期上限: {args.end}")
    print(f"CSV 資料夾: {csv_dir.resolve()}")
    print(f"缺檔回補: {'啟用，可能從 ' + args.start + ' 回補' if args.backfill_missing else '停用，日常只補既有 CSV 最後日期之後'}")
    print(f"HTML 入口: {(Path(args.output_dir) / 'html' / 'index.html').resolve()}")
    print(f"最終股票雷達: {(Path(args.output_dir) / args.final_output).resolve()}")
    print()

    update_cmd = [
        sys.executable,
        "update_prices_incremental.py",
        "--stock-file",
        args.stock_file,
        "--csv-dir",
        args.csv_dir,
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

    # 只有明確要求缺檔回補時，才傳 --start 與 --backfill-missing。
    # 日常更新不傳 --start，避免 log 看起來像從 2023 開始跑，也避免誤回補。
    if args.backfill_missing:
        update_cmd.extend(["--start", args.start, "--backfill-missing"])

    if args.max_stocks:
        update_cmd.extend(["--max-stocks", str(args.max_stocks)])

    try:
        run_cmd(update_cmd, log_path)

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                "train_lightgbm_5d.py",
                "--input",
                str(csv_dir / "all_price.csv"),
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

        if not args.skip_risk_filter:
            risk_cmd = [
                sys.executable,
                "apply_risk_filter.py",
                "--prediction",
                str(Path(args.output_dir) / "prediction_5d_lightgbm.csv"),
                "--risk-watchlist",
                args.risk_watchlist,
                "--output",
                str(Path(args.output_dir) / args.final_output),
                "--blocked-output",
                str(Path(args.output_dir) / "blocked_stock_radar.csv"),
                "--today",
                args.end,
            ]
            run_cmd(risk_cmd, log_path)

        if not args.skip_html:
            html_cmd = [
                sys.executable,
                "csv_to_html.py",
                "--csv-dir",
                args.output_dir,
                "--output-dir",
                str(Path(args.output_dir) / "html"),
                "--max-rows",
                "0",
                "--include",
                "final*.csv",
                "blocked*.csv",
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
    print(f"最終股票雷達: {(Path(args.output_dir) / args.final_output).resolve()}")


if __name__ == "__main__":
    main()
