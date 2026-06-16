# -*- coding: utf-8 -*-
"""把 TWSTOCK 產生的 CSV 轉成手機友善 HTML。

功能：
  - 讀取目前資料夾的 .csv
  - 轉成 html/*.html
  - 產生 html/index.html 入口頁
  - 支援手機版樣式、搜尋框、sticky 表頭、橫向捲動
  - prediction_*.csv 會自動依看漲機率排序
  - *_price.csv 會顯示最新收盤資訊與簡單走勢圖

安裝：
    pip install pandas numpy

基本用法：
    python csv_to_html.py

指定資料夾：
    python csv_to_html.py --csv-dir . --output-dir html

只轉預測報表：
    python csv_to_html.py --include "prediction*.csv" "backtest*.csv" "feature_importance*.csv"

轉全部 CSV，但每個 HTML 最多顯示 500 筆：
    python csv_to_html.py --max-rows 500

轉完整資料，不截斷：
    python csv_to_html.py --max-rows 0
"""

from __future__ import annotations

import argparse
import html
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


DEFAULT_INCLUDE = [
    "prediction*.csv",
    "backtest*.csv",
    "feature_importance*.csv",
    "all_price.csv",
    "*_price.csv",
]

DEFAULT_EXCLUDE_PREFIXES = (
    "failed_",
    "stocks_meta",
)


CSS = r"""
:root {
  --bg: #f6f7fb;
  --card: #ffffff;
  --text: #172033;
  --muted: #657084;
  --border: #e5e7eb;
  --accent: #2563eb;
  --up: #b91c1c;
  --down: #047857;
  --neutral: #6b7280;
  --shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC",
    "PingFang TC", "Microsoft JhengHei", Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
}
.container {
  max-width: 1180px;
  margin: 0 auto;
  padding: 16px;
}
.header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(246, 247, 251, 0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}
.header-inner {
  max-width: 1180px;
  margin: 0 auto;
  padding: 12px 16px;
}
h1 {
  font-size: 22px;
  margin: 0 0 4px;
}
h2 {
  font-size: 18px;
  margin: 24px 0 12px;
}
.meta {
  color: var(--muted);
  font-size: 13px;
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
  margin: 14px 0;
  box-shadow: var(--shadow);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
}
.stat {
  background: #f9fafb;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
}
.stat .label {
  color: var(--muted);
  font-size: 12px;
}
.stat .value {
  font-weight: 700;
  font-size: 18px;
  margin-top: 2px;
}
.search {
  width: 100%;
  padding: 12px 14px;
  border-radius: 12px;
  border: 1px solid var(--border);
  font-size: 16px;
  background: white;
}
.table-wrap {
  overflow-x: auto;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: white;
  max-height: calc(100vh - 190px);
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
  white-space: nowrap;
}
th, td {
  border-bottom: 1px solid var(--border);
  padding: 8px 10px;
  text-align: right;
}
th:first-child, td:first-child,
th.left, td.left {
  text-align: left;
}
th {
  position: sticky;
  top: 0;
  z-index: 5;
  background: #f3f4f6;
  color: #374151;
  font-weight: 700;
}
tr:hover td { background: #f9fafb; }
a {
  color: var(--accent);
  text-decoration: none;
  font-weight: 600;
}
a:hover { text-decoration: underline; }
.badge {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
}
.badge-up { background: #fee2e2; color: var(--up); }
.badge-down { background: #dcfce7; color: var(--down); }
.badge-neutral { background: #f3f4f6; color: var(--neutral); }
.num-up { color: var(--up); font-weight: 700; }
.num-down { color: var(--down); font-weight: 700; }
.footer {
  color: var(--muted);
  font-size: 12px;
  margin: 28px 0 12px;
}
.sparkline {
  width: 100%;
  max-width: 680px;
  height: 150px;
  display: block;
}
.small {
  font-size: 12px;
  color: var(--muted);
}
.file-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.file-list li {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
  border-bottom: 1px solid var(--border);
  padding: 10px 0;
}
.file-list li:last-child { border-bottom: 0; }
@media (max-width: 640px) {
  .container { padding: 10px; }
  .header-inner { padding: 10px; }
  h1 { font-size: 19px; }
  table { font-size: 12px; }
  th, td { padding: 7px 8px; }
  .card { border-radius: 14px; padding: 12px; }
  .table-wrap { max-height: calc(100vh - 170px); }
}
"""


JS = r"""
function filterTable(inputId, tableId) {
  const input = document.getElementById(inputId);
  const table = document.getElementById(tableId);
  if (!input || !table) return;

  input.addEventListener("input", function() {
    const q = input.value.toLowerCase();
    const rows = table.querySelectorAll("tbody tr");
    rows.forEach(row => {
      row.style.display = row.innerText.toLowerCase().includes(q) ? "" : "none";
    });
  });
}
document.addEventListener("DOMContentLoaded", function() {
  filterTable("search", "data-table");
});
"""



COLUMN_LABELS_ZH = {
    "stock_id": "股票代碼",
    "date": "資料日期",
    "close": "收盤價",
    "prediction": "預測結果",
    "prediction_5d": "預測5日漲跌",
    "prob_up": "看漲機率",
    "prob_down": "看跌機率",
    "prob_up_5d": "5日看漲機率",
    "prob_down_5d": "5日看跌機率",
    "ret_1": "近1日漲跌幅",
    "ret_2": "近2日漲跌幅",
    "ret_3": "近3日漲跌幅",
    "ret_5": "近5日漲跌幅",
    "ret_10": "近10日漲跌幅",
    "ret_20": "近20日漲跌幅",
    "ma5_gap": "收盤價偏離5日均線",
    "ma10_gap": "收盤價偏離10日均線",
    "ma20_gap": "收盤價偏離20日均線",
    "ma60_gap": "收盤價偏離60日均線",
    "ma5_ma20_gap": "5日均線偏離20日均線",
    "ma20_ma60_gap": "20日均線偏離60日均線",
    "rsi_14": "RSI 14日",
    "volume_ratio_5": "成交量/5日均量",
    "volume_ratio_20": "成交量/20日均量",
    "volume_trend_5_20": "5日均量/20日均量",
    "future_return_5d": "未來5日實際報酬",
    "target_5d_up": "未來5日實際是否上漲",
    "signal_score": "技術分數",
    "confidence": "信心程度",
    "backtest_hit_rate": "歷史命中率",
    "backtest_samples": "回測樣本數",
    "split": "資料切分",
    "samples": "樣本數",
    "positive_rate": "上漲樣本比例",
    "accuracy": "準確率",
    "precision": "精確率",
    "recall": "召回率",
    "f1": "F1分數",
    "auc": "AUC",
    "feature": "特徵名稱",
    "importance_gain": "重要度 Gain",
    "importance_split": "重要度 Split",
    "avg_prob_up": "平均看漲機率",
    "avg_future_5d_return": "平均未來5日報酬",
    "median_future_5d_return": "未來5日報酬中位數",
    "hit_rate": "上漲命中率",
    "selected_count": "選取檔數",
    "stock_ids": "股票清單",
    "reason": "判斷依據",
}

def col_label(col: object) -> str:
    col_str = str(col)
    return COLUMN_LABELS_ZH.get(col_str, col_str)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def esc(value: object) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def format_value(value: object, col: str = "") -> str:
    if pd.isna(value):
        return ""

    col_lower = str(col).lower()

    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"

    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return ""
        # 機率、比例、報酬率欄位用百分比
        if any(key in col_lower for key in ["prob", "return", "ret", "rate", "gap", "ratio", "auc", "accuracy", "precision", "recall", "f1"]):
            return f"{float(value) * 100:.2f}%"
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}"
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")

    return esc(value)


def class_for_cell(value: object, col: str = "") -> str:
    col_lower = str(col).lower()
    try:
        num = float(value)
    except Exception:
        num = np.nan

    if any(key in col_lower for key in ["return", "ret", "spread", "gap"]):
        if pd.notna(num) and num > 0:
            return "num-up"
        if pd.notna(num) and num < 0:
            return "num-down"

    return ""


def badge_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    if text in ["漲", "up", "UP", "看漲"]:
        return f'<span class="badge badge-up">{esc(text)}</span>'
    if text in ["跌", "down", "DOWN", "看跌"]:
        return f'<span class="badge badge-down">{esc(text)}</span>'
    if text in ["中性", "資料不足"]:
        return f'<span class="badge badge-neutral">{esc(text)}</span>'
    return esc(text)


def dataframe_to_html_table(df: pd.DataFrame, table_id: str = "data-table") -> str:
    if df.empty:
        return "<p class='small'>沒有資料</p>"

    left_cols = {"stock_id", "date", "prediction", "prediction_5d", "confidence", "reason", "split", "feature", "stock_ids"}

    thead = "<thead><tr>"
    for col in df.columns:
        cls = "left" if str(col) in left_cols else ""
        thead += f'<th class="{cls}" title="{esc(col)}">{esc(col_label(col))}</th>'
    thead += "</tr></thead>"

    rows = []
    for _, row in df.iterrows():
        tds = []
        for col in df.columns:
            value = row[col]
            cls = class_for_cell(value, col)
            if str(col) in left_cols:
                cls = (cls + " left").strip()

            if str(col) in {"prediction", "prediction_5d"}:
                rendered = badge_text(value)
            else:
                rendered = format_value(value, col)

            tds.append(f'<td class="{cls}">{rendered}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")

    return f'<div class="table-wrap"><table id="{table_id}">{thead}<tbody>' + "\n".join(rows) + "</tbody></table></div>"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"stock_id": str})


def sort_for_view(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()

    # 日期欄位盡量轉成 yyyy-mm-dd
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if dt.notna().any():
            df["date"] = dt.dt.strftime("%Y-%m-%d")

    name_lower = name.lower()

    # 預測檔：依看漲機率排序
    for col in ["prob_up_5d", "prob_up", "avg_prob_up"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values(col, ascending=False)
            return df

    # 特徵重要度
    if "importance_gain" in df.columns:
        df["importance_gain"] = pd.to_numeric(df["importance_gain"], errors="coerce")
        return df.sort_values("importance_gain", ascending=False)

    # 個股價格：新到舊
    if "date" in df.columns and name_lower.endswith("_price.csv"):
        return df.sort_values("date", ascending=False)

    return df


def limit_rows(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    if max_rows and max_rows > 0 and len(df) > max_rows:
        return df.head(max_rows).copy(), True
    return df, False


def sparkline_svg(df: pd.DataFrame) -> str:
    if "close" not in df.columns or "date" not in df.columns:
        return ""

    tmp = df.copy()
    tmp["date_dt"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp["close_num"] = pd.to_numeric(tmp["close"], errors="coerce")
    tmp = tmp.dropna(subset=["date_dt", "close_num"]).sort_values("date_dt").tail(80)

    if len(tmp) < 2:
        return ""

    values = tmp["close_num"].to_numpy(dtype=float)
    width, height = 680, 150
    pad = 12
    min_v, max_v = float(np.min(values)), float(np.max(values))
    if max_v == min_v:
        max_v += 1.0
        min_v -= 1.0

    points = []
    for i, v in enumerate(values):
        x = pad + i * (width - 2 * pad) / (len(values) - 1)
        y = height - pad - (v - min_v) * (height - 2 * pad) / (max_v - min_v)
        points.append(f"{x:.2f},{y:.2f}")

    first = values[0]
    last = values[-1]
    cls_color = "#b91c1c" if last >= first else "#047857"

    return f"""
<div class="card">
  <div class="meta">最近 {len(tmp)} 筆收盤走勢</div>
  <svg class="sparkline" viewBox="0 0 {width} {height}" role="img" aria-label="close price sparkline">
    <polyline fill="none" stroke="{cls_color}" stroke-width="3" points="{' '.join(points)}"></polyline>
    <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#e5e7eb" stroke-width="1"></line>
  </svg>
  <div class="small">Start {first:.2f} → Last {last:.2f}</div>
</div>
"""


def summary_cards(df: pd.DataFrame, file_name: str, truncated: bool, original_rows: int) -> str:
    cards = []

    def card(label: str, value: object) -> str:
        return f'<div class="stat"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div></div>'

    cards.append(card("資料列數", f"{original_rows:,}" + (" / truncated" if truncated else "")))
    cards.append(card("欄位數", f"{len(df.columns):,}"))

    if "stock_id" in df.columns:
        cards.append(card("股票數", f"{df['stock_id'].nunique():,}"))

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.notna().any():
            cards.append(card("日期區間", f"{dates.min().date()} ~ {dates.max().date()}"))

    if "prediction_5d" in df.columns:
        counts = df["prediction_5d"].value_counts(dropna=False).to_dict()
        cards.append(card("5日預測分布", " / ".join(f"{k}:{v}" for k, v in counts.items())))

    if "prediction" in df.columns:
        counts = df["prediction"].value_counts(dropna=False).to_dict()
        cards.append(card("預測分布", " / ".join(f"{k}:{v}" for k, v in counts.items())))

    return '<div class="grid">' + "\n".join(cards) + "</div>"


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
<script>{JS}</script>
</head>
<body>
<div class="header">
  <div class="header-inner">
    <h1>{esc(title)}</h1>
    <div class="meta">Generated at {esc(now_text())}</div>
  </div>
</div>
<div class="container">
{body}
<div class="footer">Generated by csv_to_html.py</div>
</div>
</body>
</html>
"""


def make_html_for_csv(csv_path: Path, html_path: Path, max_rows: int) -> dict:
    df_raw = read_csv(csv_path)
    original_rows = len(df_raw)
    df_sorted = sort_for_view(df_raw, csv_path.name)
    df_view, truncated = limit_rows(df_sorted, max_rows)

    body_parts = [
        '<p><a href="index.html">← 回 index</a></p>',
        '<div class="card">',
        summary_cards(df_sorted, csv_path.name, truncated, original_rows),
        '</div>',
    ]

    if csv_path.name.lower().endswith("_price.csv") and csv_path.name.lower() != "all_price.csv":
        body_parts.append(sparkline_svg(df_sorted))

    if truncated:
        body_parts.append(
            f'<div class="card small">注意：原始 CSV 有 {original_rows:,} 筆，'
            f'此 HTML 只顯示前 {len(df_view):,} 筆。若要完整顯示請用 --max-rows 0。</div>'
        )

    body_parts.append('<div class="card"><input id="search" class="search" placeholder="搜尋表格內容，例如股票代碼、日期、漲跌..."></div>')
    body_parts.append(dataframe_to_html_table(df_view))

    html_path.write_text(page(csv_path.name, "\n".join(body_parts)), encoding="utf-8")

    return {
        "csv": csv_path.name,
        "html": html_path.name,
        "rows": original_rows,
        "cols": len(df_raw.columns),
        "modified": datetime.fromtimestamp(csv_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def collect_csv_files(csv_dir: Path, include_patterns: list[str], exclude_prefixes: tuple[str, ...]) -> list[Path]:
    files = []
    for pattern in include_patterns:
        files.extend(csv_dir.glob(pattern))

    unique = []
    seen = set()
    for path in files:
        if not path.is_file():
            continue
        name = path.name
        lower = name.lower()
        if not lower.endswith(".csv"):
            continue
        if any(lower.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        unique.append(path)

    priority = {
        "prediction_5d_lightgbm.csv": 0,
        "prediction_tomorrow.csv": 1,
        "backtest_report.csv": 2,
        "backtest_daily_topN.csv": 3,
        "feature_importance_lightgbm.csv": 4,
        "all_price.csv": 5,
    }

    return sorted(unique, key=lambda p: (priority.get(p.name, 100), p.name))


def make_index(entries: list[dict], output_dir: Path) -> None:
    body = [
        '<div class="card">',
        '<p>手機打開這個 <b>index.html</b> 後，可以點下面的 HTML 報表查看內容。</p>',
        '<input id="search" class="search" placeholder="搜尋檔名，例如 prediction、2330、backtest...">',
        '</div>',
        '<div class="card">',
        '<ul class="file-list" id="data-table">',
    ]

    for item in entries:
        body.append(
            "<li>"
            f'<span><a href="{esc(item["html"])}">{esc(item["csv"])}</a>'
            f'<div class="small">{item["rows"]:,} rows · {item["cols"]} cols · {esc(item["modified"])}</div></span>'
            f'<span class="small">{esc(item["html"])}</span>'
            "</li>"
        )

    body.extend(["</ul>", "</div>"])

    # index 的搜尋不是 table，補一段可搜尋 li 的 script
    index_html = page("TWSTOCK HTML Reports", "\n".join(body))
    index_html = index_html.replace(
        "</script>",
        r"""
document.addEventListener("DOMContentLoaded", function() {
  const input = document.getElementById("search");
  const list = document.getElementById("data-table");
  if (!input || !list) return;
  input.addEventListener("input", function() {
    const q = input.value.toLowerCase();
    list.querySelectorAll("li").forEach(li => {
      li.style.display = li.innerText.toLowerCase().includes(q) ? "" : "none";
    });
  });
});
</script>""",
        1,
    )

    (output_dir / "index.html").write_text(index_html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert TWSTOCK CSV files to mobile-friendly HTML.")
    parser.add_argument("--csv-dir", default=".", help="CSV 所在資料夾，預設目前資料夾。")
    parser.add_argument("--output-dir", default="html", help="HTML 輸出資料夾，預設 html。")
    parser.add_argument(
        "--include",
        nargs="*",
        default=DEFAULT_INCLUDE,
        help="要轉換的 glob pattern。預設包含 prediction/backtest/feature/all_price/*_price。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="每個 HTML 最多顯示幾筆。0 表示不限制。預設 500。",
    )
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="預設會略過 failed_*.csv；加這個參數會保留。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_prefixes = tuple() if args.keep_failed else DEFAULT_EXCLUDE_PREFIXES
    csv_files = collect_csv_files(csv_dir, args.include, exclude_prefixes)

    if not csv_files:
        raise RuntimeError(f"找不到 CSV：{csv_dir.resolve()}")

    print(f"找到 {len(csv_files)} 個 CSV")
    print(f"輸出資料夾: {output_dir.resolve()}")

    entries = []
    for i, csv_path in enumerate(csv_files, start=1):
        html_path = output_dir / f"{csv_path.stem}.html"
        try:
            entry = make_html_for_csv(csv_path, html_path, max_rows=args.max_rows)
            entries.append(entry)
            print(f"[{i}/{len(csv_files)}] {csv_path.name} -> {html_path}")
        except Exception as exc:
            print(f"[{i}/{len(csv_files)}] 失敗 {csv_path.name}: {exc}")

    make_index(entries, output_dir)

    print()
    print(f"完成。手機請打開：{(output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
