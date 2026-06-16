# -*- coding: utf-8 -*-
"""TWSTOCK 一鍵流程：更新股票清單、抓 CSV、訓練 5 日漲跌、轉 HTML。

放在 TWSTOCK 資料夾後，只要執行：

    python run_all.py

預設會做：
  1. 更新 stocks.txt
  2. 強制重新抓所有股票/ETF 價量 CSV，並產生 all_price.csv
  3. 用 all_price.csv 訓練 LightGBM 未來 5 日漲跌模型
  4. 把預測/回測報表轉成手機方便看的 HTML
  5. 入口頁在 html/index.html

第一次執行前請確認已安裝：
    pip install pandas numpy requests lxml html5lib lightgbm

快速測試前 20 檔：
    python run_all.py --max-stocks 20

續跑，不重抓已存在的個股 CSV：
    python run_all.py --resume

只更新資料與 HTML，不訓練模型：
    python run_all.py --skip-train

轉換所有個股 CSV 成 HTML：
    python run_all.py --html-all-csv

注意：
    預設 --force 會重新抓指定日期區間的所有股票/ETF，2285 檔會跑很久。
    若只是中斷後續跑，請使用 --resume。
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional


REQUIRED_FILES = {
    "stock_list": "make_stocks_txt.py",
    "fetch": "example.py",
    "train": "train_lightgbm_5d.py",
    "html": "csv_to_html.py",
}

LOG_FILE = "pipeline_log.txt"


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)


def run_cmd(cmd: List[str], log_path: Path, dry_run: bool = False) -> None:
    header = "\n" + "=" * 90 + "\n"
    header += f"[{now()}] RUN: {quote_cmd(cmd)}\n"
    header += "=" * 90 + "\n"

    print(header, end="")
    write_log(log_path, header)

    if dry_run:
        return

    process = subprocess.Popen(
        cmd,
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        write_log(log_path, line)

    return_code = process.wait()

    footer = f"\n[{now()}] EXIT CODE: {return_code}\n"
    print(footer, end="")
    write_log(log_path, footer)

    if return_code != 0:
        raise RuntimeError(f"命令失敗，exit code={return_code}: {quote_cmd(cmd)}")


def check_required_files(args: argparse.Namespace) -> None:
    missing = []

    if not args.skip_stock_list:
        if not Path(REQUIRED_FILES["stock_list"]).exists():
            missing.append(REQUIRED_FILES["stock_list"])

    if not args.skip_fetch:
        if not Path(REQUIRED_FILES["fetch"]).exists():
            missing.append(REQUIRED_FILES["fetch"])
        if not Path("local_data_loader.py").exists():
            missing.append("local_data_loader.py")

    if not args.skip_train:
        if not Path(REQUIRED_FILES["train"]).exists():
            missing.append(REQUIRED_FILES["train"])

    if not args.skip_html:
        if not Path(REQUIRED_FILES["html"]).exists():
            missing.append(REQUIRED_FILES["html"])

    if missing:
        msg = "缺少必要檔案：\n" + "\n".join(f"  - {x}" for x in missing)
        msg += "\n\n請確認這些檔案都放在同一個 TWSTOCK 資料夾。"
        raise FileNotFoundError(msg)


def check_python_packages(args: argparse.Namespace) -> None:
    """做最基本 package 檢查，避免跑到一半才爆。"""
    required = ["pandas", "numpy", "requests"]
    if not args.skip_train:
        required.append("lightgbm")

    missing = []
    for package in required:
        try:
            __import__(package)
        except ModuleNotFoundError:
            missing.append(package)

    if missing:
        print("缺少 Python 套件：")
        for package in missing:
            print(f"  - {package}")
        print()
        print("請先安裝：")
        print("  pip install pandas numpy requests lxml html5lib lightgbm")
        raise SystemExit(1)


def backup_existing_outputs(output_dir: Path, backup: bool) -> Optional[Path]:
    if not backup:
        return None

    targets = [
        "all_price.csv",
        "prediction_5d_lightgbm.csv",
        "backtest_report.csv",
        "backtest_daily_topN.csv",
        "backtest_topN_summary.json",
        "feature_importance_lightgbm.csv",
        "lightgbm_5d_model.txt",
    ]

    existing = [Path(x) for x in targets if Path(x).exists()]
    html_dir = output_dir / "html"
    if html_dir.exists():
        existing.append(html_dir)

    if not existing:
        return None

    backup_dir = output_dir / "backup" / time.strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for path in existing:
        dest = backup_dir / path.name
        try:
            path.replace(dest)
        except Exception:
            # 若 replace 因跨磁碟/資料夾問題失敗，不中斷主流程。
            pass

    return backup_dir


def build_stock_list_cmd(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        REQUIRED_FILES["stock_list"],
        "--output",
        args.stock_file,
        "--meta",
        args.stock_meta,
    ]


def build_fetch_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        REQUIRED_FILES["fetch"],
        "--stock-file",
        args.stock_file,
        "--start",
        args.start,
        "--end",
        args.end,
        "--out-dir",
        args.csv_dir,
        "--sleep",
        str(args.sleep),
        "--retries",
        str(args.retries),
        "--retry-sleep",
        str(args.retry_sleep),
        "--combined",
    ]

    if not args.resume:
        cmd.append("--force")

    if args.max_stocks and args.max_stocks > 0:
        cmd.extend(["--max-stocks", str(args.max_stocks)])

    if args.show_tail:
        cmd.append("--show-tail")

    return cmd


def build_train_cmd(args: argparse.Namespace) -> List[str]:
    input_path = Path(args.csv_dir) / "all_price.csv"

    cmd = [
        sys.executable,
        REQUIRED_FILES["train"],
        "--input",
        str(input_path),
        "--output-dir",
        args.output_dir,
        "--top-n",
        str(args.top_n),
        "--horizon",
        str(args.horizon),
        "--target-return",
        str(args.target_return),
        "--min-history",
        str(args.min_history),
        "--num-boost-round",
        str(args.num_boost_round),
        "--early-stopping-rounds",
        str(args.early_stopping_rounds),
    ]

    if args.is_unbalance:
        cmd.append("--is-unbalance")

    return cmd


def build_html_cmd(args: argparse.Namespace) -> List[str]:
    html_dir = str(Path(args.output_dir) / "html")

    cmd = [
        sys.executable,
        REQUIRED_FILES["html"],
        "--csv-dir",
        args.output_dir,
        "--output-dir",
        html_dir,
        "--max-rows",
        str(args.html_max_rows),
    ]

    if args.html_all_csv or args.skip_train:
        # 全部轉，包含 all_price.csv 與所有 *_price.csv。
        # 注意：若使用 --skip-train，就不會有 prediction/backtest CSV，
        # 所以必須把 all_price.csv / *_price.csv 納入，否則 HTML 步驟會找不到檔案。
        cmd.extend(
            [
                "--include",
                "prediction*.csv",
                "backtest*.csv",
                "feature_importance*.csv",
                "all_price.csv",
                "*_price.csv",
            ]
        )
    else:
        # 預設只轉預測與回測報表，手機最實用也最快。
        cmd.extend(
            [
                "--include",
                "prediction*.csv",
                "backtest*.csv",
                "feature_importance*.csv",
            ]
        )

    return cmd


def print_summary(args: argparse.Namespace) -> None:
    print("TWSTOCK 一鍵流程")
    print(f"工作資料夾: {Path.cwd()}")
    print(f"股票清單: {args.stock_file}")
    print(f"日期區間: {args.start} ~ {args.end}")
    print(f"CSV 輸出: {Path(args.csv_dir).resolve()}")
    print(f"報表輸出: {Path(args.output_dir).resolve()}")
    print(f"HTML 入口: {(Path(args.output_dir) / 'html' / 'index.html').resolve()}")
    print(f"抓資料模式: {'續跑，跳過已存在 CSV' if args.resume else '強制刷新既有 CSV'}")
    print(f"sleep/retries/retry-sleep: {args.sleep}/{args.retries}/{args.retry_sleep}")
    if args.max_stocks:
        print(f"測試模式：只處理前 {args.max_stocks} 檔")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full TWSTOCK pipeline: fetch CSV -> train LightGBM -> export HTML.")

    parser.add_argument("--stock-file", default="stocks.txt", help="股票清單檔，預設 stocks.txt")
    parser.add_argument("--stock-meta", default="stocks_meta.csv", help="股票清單 metadata，預設 stocks_meta.csv")
    parser.add_argument("--start", default="2023-01-01", help="抓資料起始日期，預設 2023-01-01")
    parser.add_argument("--end", default=date.today().isoformat(), help="抓資料結束日期，預設今天")
    parser.add_argument("--csv-dir", default=".", help="個股 CSV 與 all_price.csv 輸出資料夾，預設目前資料夾")
    parser.add_argument("--output-dir", default=".", help="模型/預測/HTML 輸出資料夾，預設目前資料夾")

    parser.add_argument("--sleep", type=float, default=3.0, help="抓資料 request 間隔秒數，預設 3")
    parser.add_argument("--retries", type=int, default=8, help="抓資料單次 request 重試次數，預設 8")
    parser.add_argument("--retry-sleep", type=float, default=20.0, help="第一次重試等待秒數，預設 20")
    parser.add_argument("--resume", action="store_true", help="續跑模式：已存在的 *_price.csv 會跳過，不會強制重抓")
    parser.add_argument("--max-stocks", type=int, default=0, help="只處理前 N 檔，方便測試；0 表示不限制")
    parser.add_argument("--show-tail", action="store_true", help="抓資料成功後顯示每檔最後 5 筆，通常不建議大量批次使用")

    parser.add_argument("--top-n", type=int, default=50, help="LightGBM 回測每日挑前 N 檔，預設 50")
    parser.add_argument("--horizon", type=int, default=5, help="預測未來幾個交易日，預設 5")
    parser.add_argument("--target-return", type=float, default=0.0, help="未來報酬大於此值才算漲，預設 0")
    parser.add_argument("--min-history", type=int, default=60, help="每檔至少需要幾筆資料才訓練，預設 60")
    parser.add_argument("--num-boost-round", type=int, default=800, help="LightGBM num_boost_round，預設 800")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="LightGBM early stopping，預設 50")
    parser.add_argument("--is-unbalance", action="store_true", help="LightGBM 類別不平衡處理")

    parser.add_argument("--html-max-rows", type=int, default=500, help="每個 HTML 最多顯示幾筆，0 表示不限制，預設 500")
    parser.add_argument("--html-all-csv", action="store_true", help="把所有 *_price.csv 也轉成 HTML；預設只轉預測/回測報表")

    parser.add_argument("--skip-stock-list", action="store_true", help="跳過更新 stocks.txt")
    parser.add_argument("--skip-fetch", action="store_true", help="跳過抓 CSV")
    parser.add_argument("--skip-train", action="store_true", help="跳過訓練 LightGBM")
    parser.add_argument("--skip-html", action="store_true", help="跳過轉 HTML")
    parser.add_argument("--backup", action="store_true", help="執行前把舊報表/模型/html 移到 backup/YYYYmmdd_HHMMSS")
    parser.add_argument("--dry-run", action="store_true", help="只印出會執行的命令，不真的執行")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    log_path = Path(LOG_FILE)
    log_path.write_text(f"TWSTOCK pipeline started at {now()}\n", encoding="utf-8")

    print_summary(args)
    check_required_files(args)
    check_python_packages(args)

    if args.backup:
        backup_dir = backup_existing_outputs(Path(args.output_dir), backup=True)
        if backup_dir:
            print(f"舊輸出已移到: {backup_dir.resolve()}")
            print()

    started = time.time()

    try:
        if not args.skip_stock_list:
            run_cmd(build_stock_list_cmd(args), log_path=log_path, dry_run=args.dry_run)

        if not args.skip_fetch:
            run_cmd(build_fetch_cmd(args), log_path=log_path, dry_run=args.dry_run)

        if not args.skip_train:
            all_price = Path(args.csv_dir) / "all_price.csv"
            if not args.dry_run and not all_price.exists():
                raise FileNotFoundError(f"找不到 {all_price}，無法訓練。請確認抓資料步驟有加 --combined 並成功產生 all_price.csv。")
            run_cmd(build_train_cmd(args), log_path=log_path, dry_run=args.dry_run)

        if not args.skip_html:
            run_cmd(build_html_cmd(args), log_path=log_path, dry_run=args.dry_run)

    except Exception as exc:
        print()
        print("流程失敗：", exc)
        print(f"詳細 log：{log_path.resolve()}")
        raise SystemExit(1)

    elapsed = time.time() - started
    print()
    print("流程完成")
    print(f"耗時: {elapsed / 60:.1f} 分鐘")
    print(f"詳細 log: {log_path.resolve()}")

    if not args.skip_html:
        print(f"手機入口: {(Path(args.output_dir) / 'html' / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
