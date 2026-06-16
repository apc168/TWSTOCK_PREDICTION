# -*- coding: utf-8 -*-
"""?葫??啗瞍脰?嚗???*_price.csv ??all_price.csv嚗撓??prediction_tomorrow.csv??

?銝??銵閬? + 蝪⊥?甇瑕?葫???穿?銝?閬?scikit-learn嚗?
?芷?閬?pandas / numpy??

?冽? 1嚗? all_price.csv
    python predict_tomorrow.py --input all_price.csv

?冽? 2嚗???all_price.csv嚗?交?????冗??*_price.csv
    python predict_tomorrow.py --csv-dir .

?冽? 3嚗撓?箏? 100 ??瞍?
    python predict_tomorrow.py --csv-dir . --top 100

頛詨嚗?
    prediction_tomorrow.csv

????嚗?
    ???舀?鞈遣霅啜?舀?風?脣?潦?蝺??賬??賢??箇??銵????
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


EXCLUDE_PREFIXES = (
    "all_",
    "failed_",
    "prediction_",
    "stocks_meta",
)


def sigmoid(x: float) -> float:
    x = max(min(float(x), 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-x))


def clamp_series(s: pd.Series, lower: float = -1.0, upper: float = 1.0) -> pd.Series:
    return s.clip(lower=lower, upper=upper)


def clamp_value(x: float, lower: float = -1.0, upper: float = 1.0) -> float:
    if pd.isna(x):
        return 0.0
    return max(lower, min(float(x), upper))


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize local data loader output columns."""
    df = df.copy()

    # 撣貉?甈??迂撠?
    rename_map = {}
    for col in df.columns:
        lower = str(col).strip().lower()
        if lower == "trading_volume":
            rename_map[col] = "Trading_Volume"
        elif lower == "trading_money":
            rename_map[col] = "Trading_money"
        elif lower == "trading_turnover":
            rename_map[col] = "Trading_turnover"
        elif lower == "stockid":
            rename_map[col] = "stock_id"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError("CSV 蝻箏? date 甈?")
    if "stock_id" not in df.columns:
        # ?格? csv ?交???stock_id嚗停敺???嚗?怎垢????
        df["stock_id"] = ""
    if "close" not in df.columns:
        raise ValueError("CSV 蝻箏? close 甈?")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["stock_id"] = df["stock_id"].astype(str).str.strip()

    for col in ["open", "max", "min", "close", "spread", "Trading_Volume", "Trading_money", "Trading_turnover", "ma5", "ma20"]:
        if col in df.columns:
            df[col] = safe_num(df[col])

    return df


def stock_id_from_filename(path: Path) -> str:
    name = path.name
    if name.endswith("_price.csv"):
        return name[: -len("_price.csv")]
    return path.stem


def should_skip_csv(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".csv"):
        return True
    if not name.endswith("_price.csv"):
        return True
    return any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES)


def load_prices(input_file: Optional[str], csv_dir: str) -> pd.DataFrame:
    if input_file:
        path = Path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"?曆??啗撓?交?: {path}")
        df = pd.read_csv(path, dtype={"stock_id": str})
        return normalize_columns(df)

    folder = Path(csv_dir)
    if not folder.exists():
        raise FileNotFoundError(f"?曆??啗??冗: {folder}")

    frames: list[pd.DataFrame] = []
    for path in sorted(folder.glob("*_price.csv")):
        if should_skip_csv(path):
            continue
        try:
            one = pd.read_csv(path, dtype={"stock_id": str})
            one = normalize_columns(one)
            if one["stock_id"].eq("").all():
                one["stock_id"] = stock_id_from_filename(path)
            frames.append(one)
        except Exception as exc:
            print(f"?仿? {path.name}: {exc}")

    if not frames:
        raise RuntimeError(f"??{folder.resolve()} ?曆??啣?函? *_price.csv")

    return pd.concat(frames, ignore_index=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["close"] = safe_num(df["close"])
    if "open" in df.columns:
        df["open"] = safe_num(df["open"])
    else:
        df["open"] = np.nan

    if "Trading_Volume" in df.columns:
        df["Trading_Volume"] = safe_num(df["Trading_Volume"])
    else:
        df["Trading_Volume"] = np.nan

    df["ma5"] = df["close"].rolling(5, min_periods=5).mean()
    df["ma10"] = df["close"].rolling(10, min_periods=10).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=20).mean()
    df["ma60"] = df["close"].rolling(60, min_periods=60).mean()

    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)

    df["ma5_gap"] = df["close"] / df["ma5"] - 1.0
    df["ma20_gap"] = df["close"] / df["ma20"] - 1.0
    df["ma5_ma20_gap"] = df["ma5"] / df["ma20"] - 1.0

    df["vol5"] = df["Trading_Volume"].rolling(5, min_periods=5).mean()
    df["volume_ratio"] = df["Trading_Volume"] / df["vol5"]

    df["candle_return"] = np.where(
        df["open"].notna() & (df["open"] != 0),
        df["close"] / df["open"] - 1.0,
        np.nan,
    )

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df.loc[(loss == 0) & (gain > 0), "rsi14"] = 100.0
    df.loc[(gain == 0) & (loss > 0), "rsi14"] = 0.0

    df["next_close"] = df["close"].shift(-1)
    df["next_up"] = df["next_close"] > df["close"]

    return df


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """撠?銝憭拍??粹??亦?瞍脫??雿輻?嗅予隞亙?鞈???""
    df = df.copy()

    score = pd.Series(0.0, index=df.index)

    # ?
    score += clamp_series(df["ret1"] / 0.025) * 0.70
    score += clamp_series(df["ret3"] / 0.050) * 0.70
    score += clamp_series(df["ret5"] / 0.080) * 0.50
    score += clamp_series(df["ret10"] / 0.120) * 0.25

    # ??頞典
    score += clamp_series(df["ma5_gap"] / 0.030) * 0.65
    score += clamp_series(df["ma20_gap"] / 0.060) * 0.45
    score += clamp_series(df["ma5_ma20_gap"] / 0.040) * 0.60

    # K 璉?
    score += clamp_series(df["candle_return"] / 0.025) * 0.35

    # ?嚗?瞍脫????銝??暸????
    vol_component = (df["volume_ratio"] - 1.0).clip(lower=-1.0, upper=2.0)
    score += np.where(df["ret1"] > 0, vol_component * 0.25, -vol_component * 0.25)

    # RSI嚗?撘瑕???雿??梁????摹???嚗??漲頞都?亙?銝暺?敶?
    rsi = df["rsi14"]
    rsi_component = pd.Series(0.0, index=df.index)
    rsi_component += np.where((rsi >= 50) & (rsi <= 68), (rsi - 50) / 18 * 0.45, 0.0)
    rsi_component += np.where((rsi > 68) & (rsi <= 78), 0.20, 0.0)
    rsi_component += np.where(rsi > 78, -0.35, 0.0)
    rsi_component += np.where((rsi < 50) & (rsi >= 35), -(50 - rsi) / 15 * 0.45, 0.0)
    rsi_component += np.where(rsi < 30, 0.15, 0.0)
    score += rsi_component

    df["signal_score"] = score.replace([np.inf, -np.inf], np.nan)
    df["prob_up"] = df["signal_score"].apply(lambda x: sigmoid(x) if pd.notna(x) else np.nan)

    return df


def direction_from_prob(prob: float, up_threshold: float, down_threshold: float) -> str:
    if pd.isna(prob):
        return "鞈?銝雲"
    if prob >= up_threshold:
        return "瞍?
    if prob <= down_threshold:
        return "頝?
    return "銝剜?


def confidence_from_prob(prob: float) -> str:
    if pd.isna(prob):
        return "雿?
    distance = abs(prob - 0.5)
    if distance >= 0.20:
        return "擃?
    if distance >= 0.10:
        return "銝?
    return "雿?


def explain_last(row: pd.Series) -> str:
    reasons: list[str] = []

    def add(condition: bool, text: str) -> None:
        if condition:
            reasons.append(text)

    add(pd.notna(row.get("ma5")) and pd.notna(row.get("ma20")) and row["ma5"] > row["ma20"], "MA5>MA20")
    add(pd.notna(row.get("close")) and pd.notna(row.get("ma5")) and row["close"] > row["ma5"], "?嗥蝡?MA5")
    add(pd.notna(row.get("close")) and pd.notna(row.get("ma20")) and row["close"] > row["ma20"], "?嗥蝡?MA20")
    add(pd.notna(row.get("ret3")) and row["ret3"] > 0, "3?亙??賣迤")
    add(pd.notna(row.get("ret5")) and row["ret5"] > 0, "5?亙??賣迤")
    add(pd.notna(row.get("volume_ratio")) and row["volume_ratio"] >= 1.2, "??曉之")
    add(pd.notna(row.get("rsi14")) and row["rsi14"] >= 70, "RSI?")
    add(pd.notna(row.get("rsi14")) and row["rsi14"] <= 30, "RSI?")

    if not reasons:
        return "?⊥?憿舀?銵???

    return "??.join(reasons[:6])


def backtest_stats(scored: pd.DataFrame, up_threshold: float, down_threshold: float) -> tuple[float, int]:
    test = scored.dropna(subset=["prob_up", "next_up"]).copy()
    if test.empty:
        return np.nan, 0

    test["pred"] = np.where(
        test["prob_up"] >= up_threshold,
        True,
        np.where(test["prob_up"] <= down_threshold, False, np.nan),
    )
    test = test.dropna(subset=["pred"])

    if test.empty:
        return np.nan, 0

    hit_rate = (test["pred"].astype(bool) == test["next_up"].astype(bool)).mean()
    return float(hit_rate), int(len(test))


def predict_one_stock(
    stock_id: str,
    df: pd.DataFrame,
    min_rows: int,
    up_threshold: float,
    down_threshold: float,
) -> dict:
    one = df[df["stock_id"].astype(str) == str(stock_id)].copy()
    one = one.dropna(subset=["date", "close"]).sort_values("date")

    if len(one) < min_rows:
        last_date = one["date"].max() if not one.empty else ""
        return {
            "stock_id": stock_id,
            "last_date": last_date,
            "last_close": np.nan,
            "prediction": "鞈?銝雲",
            "prob_up": np.nan,
            "prob_down": np.nan,
            "confidence": "雿?,
            "signal_score": np.nan,
            "backtest_hit_rate": np.nan,
            "backtest_samples": 0,
            "reason": f"鞈?蝑銝雲嚗len(one)} < {min_rows}",
        }

    enriched = add_indicators(one)
    scored = score_dataframe(enriched)
    last = scored.iloc[-1]

    prob_up = float(last["prob_up"]) if pd.notna(last["prob_up"]) else np.nan
    prob_down = 1.0 - prob_up if pd.notna(prob_up) else np.nan
    prediction = direction_from_prob(prob_up, up_threshold, down_threshold)

    hit_rate, samples = backtest_stats(scored, up_threshold, down_threshold)

    return {
        "stock_id": stock_id,
        "last_date": last.get("date", ""),
        "last_close": last.get("close", np.nan),
        "prediction": prediction,
        "prob_up": round(prob_up, 4) if pd.notna(prob_up) else np.nan,
        "prob_down": round(prob_down, 4) if pd.notna(prob_down) else np.nan,
        "confidence": confidence_from_prob(prob_up),
        "signal_score": round(float(last["signal_score"]), 4) if pd.notna(last.get("signal_score")) else np.nan,
        "backtest_hit_rate": round(hit_rate, 4) if pd.notna(hit_rate) else np.nan,
        "backtest_samples": samples,
        "ret1": round(float(last.get("ret1", np.nan)), 4) if pd.notna(last.get("ret1")) else np.nan,
        "ret5": round(float(last.get("ret5", np.nan)), 4) if pd.notna(last.get("ret5")) else np.nan,
        "ma5": round(float(last.get("ma5", np.nan)), 4) if pd.notna(last.get("ma5")) else np.nan,
        "ma20": round(float(last.get("ma20", np.nan)), 4) if pd.notna(last.get("ma20")) else np.nan,
        "rsi14": round(float(last.get("rsi14", np.nan)), 2) if pd.notna(last.get("rsi14")) else np.nan,
        "volume_ratio": round(float(last.get("volume_ratio", np.nan)), 3) if pd.notna(last.get("volume_ratio")) else np.nan,
        "reason": explain_last(last),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="?葫??啗瞍脰?嚗撓??prediction_tomorrow.csv")
    parser.add_argument("--input", help="?蔥敺? all_price.csv??芣?摰?????--csv-dir ??*_price.csv??)
    parser.add_argument("--csv-dir", default=".", help="CSV 鞈?憭橘??身?桀?鞈?憭整?)
    parser.add_argument("--output", default="prediction_tomorrow.csv", help="頛詨瑼?嚗?閮?prediction_tomorrow.csv")
    parser.add_argument("--min-rows", type=int, default=60, help="瘥??喳??閬嗾蝑????身 60??)
    parser.add_argument("--up-threshold", type=float, default=0.53, help="?撞?瑼鳴??身 0.53??)
    parser.add_argument("--down-threshold", type=float, default=0.47, help="???瑼鳴??身 0.47??)
    parser.add_argument("--top", type=int, default=0, help="?芷＊蝷箇?瞍脫???擃???N 瑼?0 銵函內憿舐內????)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    prices = load_prices(args.input, args.csv_dir)
    prices = normalize_columns(prices)

    stock_ids = sorted(prices["stock_id"].dropna().astype(str).unique())
    results = []

    print(f"霈?亥????? {len(prices):,}")
    print(f"?∠巨/ETF ?賊?: {len(stock_ids):,}")
    print("???葫...")

    for i, stock_id in enumerate(stock_ids, start=1):
        if i % 200 == 0:
            print(f"  撌脰???{i}/{len(stock_ids)}")
        results.append(
            predict_one_stock(
                stock_id=stock_id,
                df=prices,
                min_rows=args.min_rows,
                up_threshold=args.up_threshold,
                down_threshold=args.down_threshold,
            )
        )

    pred = pd.DataFrame(results)

    # ??嚗??撞嚗???prob_up 擃雿?鞈?銝雲?暹?敺?
    order_map = {"瞍?: 0, "銝剜?: 1, "頝?: 2, "鞈?銝雲": 3}
    pred["_order"] = pred["prediction"].map(order_map).fillna(9)
    pred = pred.sort_values(["_order", "prob_up", "backtest_hit_rate"], ascending=[True, False, False])
    pred = pred.drop(columns=["_order"])

    output = Path(args.output)
    pred.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"撌脰撓?? {output.resolve()}")
    print()
    print("?葫??:")
    print(pred["prediction"].value_counts(dropna=False).to_string())

    valid = pred[pred["prediction"].isin(["瞍?, "頝?, "銝剜?])].copy()
    if not valid.empty:
        print()
        print("?撞璈??擃? 10 瑼?")
        cols = ["stock_id", "last_date", "last_close", "prediction", "prob_up", "confidence", "backtest_hit_rate", "reason"]
        print(valid.sort_values("prob_up", ascending=False)[cols].head(10).to_string(index=False))

    if args.top and args.top > 0:
        top_path = output.with_name(output.stem + f"_top{args.top}" + output.suffix)
        top = pred[pred["prediction"] == "瞍?].sort_values("prob_up", ascending=False).head(args.top)
        top.to_csv(top_path, index=False, encoding="utf-8-sig")
        print()
        print(f"Top {args.top} ?撞皜撌脰撓?? {top_path.resolve()}")

    print()
    print("??嚗?銵璅∪?嚗??舀?鞈遣霅堆?隢??皜研◢?扯??箸?Ｗ?瑯?)


if __name__ == "__main__":
    main()

