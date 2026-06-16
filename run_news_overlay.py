#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run news overlay after run_daily.py.

Example:
  python run_daily.py --csv-dir prices --output-dir .
  python run_news_overlay.py

Recommended for watchlist:
  python run_news_overlay.py --watchlist watchlist.txt --top-n 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n[run_news_overlay] RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", default="prediction_5d_lightgbm.csv")
    ap.add_argument("--news-output", default="news_recent.csv")
    ap.add_argument("--overlay-output", default="prediction_5d_with_news.csv")
    ap.add_argument("--html-output-dir", default="html")
    ap.add_argument("--watchlist", default=None)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--top-n", type=int, default=300, help="0 means all selected rows")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--max-news-per-stock", type=int, default=8)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--max-rows", default="0")
    args = ap.parse_args()

    if not Path(args.prediction).exists():
        raise FileNotFoundError(f"prediction file not found: {args.prediction}")

    if not args.skip_fetch:
        cmd = [
            sys.executable, "news_fetcher.py",
            "--prediction", args.prediction,
            "--output", args.news_output,
            "--days", str(args.days),
            "--top-n", str(args.top_n),
            "--sleep", str(args.sleep),
            "--max-news-per-stock", str(args.max_news_per_stock),
        ]
        if args.watchlist:
            cmd += ["--watchlist", args.watchlist]
        run(cmd)

    run([
        sys.executable, "news_sentiment_overlay.py",
        "--prediction", args.prediction,
        "--news", args.news_output,
        "--output", args.overlay_output,
    ])

    run([
        sys.executable, "csv_to_html.py",
        "--csv-dir", ".",
        "--output-dir", args.html_output_dir,
        "--include", args.overlay_output,
        "--max-rows", str(args.max_rows),
    ])

    html_path = Path(args.html_output_dir) / (Path(args.overlay_output).stem + ".html")
    print("\n[run_news_overlay] DONE")
    print(f"CSV : {args.overlay_output}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
