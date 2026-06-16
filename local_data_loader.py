# -*- coding: utf-8 -*-
"""local_data_loader — 台股公開資料本地抓取器。

直接向台灣證券交易所 (TWSE) 與櫃買中心 (TPEX) 的公開資料端點抓資料,
並整理成專案內部使用的標準欄位格式。

使用方式:

    from local_data_loader import DataLoader

    dl = DataLoader()
    df = dl.taiwan_stock_daily(stock_id="2330", start_date="2026-05-01", end_date="2026-06-11")

已實作的資料查詢:
    - taiwan_stock_info                          台股總覽(上市+上櫃)
    - taiwan_stock_daily                         日股價 (TWSE + TPEX)
    - taiwan_stock_per_pbr                       本益比/淨值比/殖利率 (上市)
    - taiwan_stock_institutional_investors       三大法人買賣超 (上市)
    - taiwan_stock_margin_purchase_short_sale    融資融券 (上市)

注意:
    - 證交所對同一 IP 有流量限制,預設每個 request 間隔 0.8 秒,請勿調太低,
      否則 IP 可能會被暫時限制連線。
    - 三大法人與融資融券是「整天全市場」的端點,查詢區間內每個交易日需要
      一個 request;查長區間請耐心等待,或先抓下來存成 csv/parquet。
"""

import datetime as dt
import time
from io import StringIO

import pandas as pd
import requests

TWSE = "https://www.twse.com.tw/rwd/zh"
TPEX = "https://www.tpex.org.tw/www/zh-tw"
ISIN = "https://isin.twse.com.tw/isin/C_public.jsp"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _roc_to_iso(s: str) -> str:
    """'113/01/02' 或 '113年01月02日' -> '2024-01-02'"""
    s = s.strip().rstrip("日").replace("年", "/").replace("月", "/")
    y, m, d = s.split("/")
    return "{:04d}-{:02d}-{:02d}".format(int(y) + 1911, int(m), int(d))


def _num(s, default=None):
    """把 '1,234'、'+5.0'、'X0.00'、'--' 之類的字串轉成數字"""
    s = str(s).replace(",", "").replace("+", "").strip()
    if s.startswith("X"):
        s = s[1:]
    if s in ("", "--", "---", "N/A", "None", "除權息", "除息", "除權"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _int(s, default=0):
    v = _num(s, default)
    return int(v) if v is not None else default


def _month_starts(start_date: str, end_date: str):
    """回傳 start~end 之間每個月的 1 號 (date 物件)"""
    start = dt.date.fromisoformat(start_date).replace(day=1)
    end = dt.date.fromisoformat(end_date)
    out = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = (cur.replace(day=28) + dt.timedelta(days=7)).replace(day=1)
    return out


def _dates(start_date: str, end_date: str):
    """回傳 start~end 的每一天 (date 物件),週末先排除"""
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    return [
        start + dt.timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + dt.timedelta(days=i)).weekday() < 5
    ]


class DataLoader:
    """台股公開資料 DataLoader，提供日線、個股資訊與市場資料查詢方法。"""

    def __init__(
        self,
        sleep: float = 1.5,
        retries: int = 5,
        retry_sleep: float = 5.0,
        backoff: float = 1.8,
    ):
        self.sleep = sleep
        self.retries = max(1, int(retries))
        self.retry_sleep = max(0.0, float(retry_sleep))
        self.backoff = max(1.0, float(backoff))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._stock_info = None

    def _get_json(self, url, params=None, timeout=30):
        """GET JSON with retry/backoff.

        TWSE/TPEX sometimes returns an empty body or a temporary HTML/rate-limit
        page instead of JSON. Calling res.json() directly then raises:
            Expecting value: line 1 column 1 (char 0)

        This wrapper retries those transient responses and raises a readable
        error only after all retries are exhausted.
        """
        last_error = None

        for attempt in range(1, self.retries + 1):
            time.sleep(self.sleep)

            try:
                res = self.session.get(url, params=params, timeout=timeout)

                if res.status_code in (429, 500, 502, 503, 504):
                    preview = res.text[:200].replace("\n", " ").strip()
                    raise RuntimeError(f"HTTP {res.status_code}: {preview}")

                res.raise_for_status()

                text = res.text or ""
                if not text.strip():
                    raise ValueError("empty response")

                try:
                    return res.json()
                except ValueError as exc:
                    preview = text[:200].replace("\n", " ").strip()
                    raise ValueError(f"non-JSON response: {preview}") from exc

            except Exception as exc:
                last_error = exc

                if attempt >= self.retries:
                    break

                wait = self.retry_sleep * (self.backoff ** (attempt - 1))
                print(
                    f"  request 失敗，準備重試 {attempt}/{self.retries - 1}: "
                    f"{exc}；等待 {wait:.1f}s"
                )
                time.sleep(wait)

                # Re-create the session after a failed response to avoid
                # keeping a bad/blocked connection alive.
                self.session.close()
                self.session = requests.Session()
                self.session.headers.update({"User-Agent": USER_AGENT})

        raise RuntimeError(f"資料來源連線或解析失敗，已重試 {self.retries} 次: {last_error}")

    # ------------------------------------------------------------------
    # 台股總覽
    # ------------------------------------------------------------------
    def taiwan_stock_info(self, timeout: int = 30) -> pd.DataFrame:
        """台股總覽 TaiwanStockInfo
        columns: industry_category, stock_id, stock_name, type
        """
        frames = []
        for mode, market in [("2", "twse"), ("4", "tpex")]:
            time.sleep(self.sleep)
            res = self.session.get(
                ISIN, params={"strMode": mode}, timeout=timeout
            )
            res.encoding = "cp950"
            table = pd.read_html(StringIO(res.text), header=0)[0]
            table.columns = [str(c).strip() for c in table.columns]
            code_name = table.columns[0]  # 有價證券代號及名稱
            in_stock_section = False
            rows = []
            for _, r in table.iterrows():
                first = str(r[code_name]).strip()
                # 區段標題列 (如「股票」、「上市認購(售)權證」) 只有第一欄有值
                if pd.isna(r.get("國際證券辨識號碼(ISIN Code)", None)) or str(
                    r.get("國際證券辨識號碼(ISIN Code)")
                ) in ("nan", first):
                    in_stock_section = first == "股票"
                    continue
                if not in_stock_section:
                    continue
                parts = first.replace("　", " ").split(" ", 1)
                if len(parts) != 2:
                    continue
                rows.append(
                    {
                        "industry_category": str(
                            r.get("產業別", "")
                        ).strip(),
                        "stock_id": parts[0].strip(),
                        "stock_name": parts[1].strip(),
                        "type": market,
                    }
                )
            frames.append(pd.DataFrame(rows))
        return pd.concat(frames, ignore_index=True)

    def _market_of(self, stock_id: str) -> str:
        if self._stock_info is None:
            self._stock_info = self.taiwan_stock_info()
        hit = self._stock_info[self._stock_info["stock_id"] == stock_id]
        if hit.empty:
            return "twse"  # 查不到就先當上市
        return hit.iloc[0]["type"]

    # ------------------------------------------------------------------
    # 日股價
    # ------------------------------------------------------------------
    def taiwan_stock_daily(
        self,
        stock_id: str,
        start_date: str,
        end_date: str = "",
        timeout: int = 30,
        **_,
    ) -> pd.DataFrame:
        """台灣股價資料表 TaiwanStockPrice
        columns: date, stock_id, Trading_Volume, Trading_money,
                 open, max, min, close, spread, Trading_turnover
        """
        end_date = end_date or str(dt.date.today())
        market = self._market_of(stock_id)
        rows = []
        for month in _month_starts(start_date, end_date):
            if market == "tpex":
                rows += self._tpex_daily_month(stock_id, month, timeout)
            else:
                rows += self._twse_daily_month(stock_id, month, timeout)
        df = pd.DataFrame(
            rows,
            columns=[
                "date",
                "stock_id",
                "Trading_Volume",
                "Trading_money",
                "open",
                "max",
                "min",
                "close",
                "spread",
                "Trading_turnover",
            ],
        )
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df.sort_values("date").reset_index(drop=True)

    def _twse_daily_month(self, stock_id, month, timeout):
        data = self._get_json(
            f"{TWSE}/afterTrading/STOCK_DAY",
            params={
                "date": month.strftime("%Y%m%d"),
                "stockNo": stock_id,
                "response": "json",
            },
            timeout=timeout,
        )
        if data.get("stat") != "OK":
            return []
        rows = []
        for d in data.get("data", []):
            rows.append(
                {
                    "date": _roc_to_iso(d[0]),
                    "stock_id": stock_id,
                    "Trading_Volume": _int(d[1]),
                    "Trading_money": _int(d[2]),
                    "open": _num(d[3]),
                    "max": _num(d[4]),
                    "min": _num(d[5]),
                    "close": _num(d[6]),
                    "spread": _num(d[7], 0.0),
                    "Trading_turnover": _int(d[8]),
                }
            )
        return rows

    def _tpex_daily_month(self, stock_id, month, timeout):
        data = self._get_json(
            f"{TPEX}/afterTrading/tradingStock",
            params={
                "code": stock_id,
                "date": month.strftime("%Y/%m/%d"),
                "response": "json",
            },
            timeout=timeout,
        )
        tables = data.get("tables") or []
        if not tables or not tables[0].get("data"):
            return []
        rows = []
        for d in tables[0]["data"]:
            # 櫃買單位是 仟股 / 仟元
            rows.append(
                {
                    "date": _roc_to_iso(d[0]),
                    "stock_id": stock_id,
                    "Trading_Volume": _int(d[1]) * 1000,
                    "Trading_money": _int(d[2]) * 1000,
                    "open": _num(d[3]),
                    "max": _num(d[4]),
                    "min": _num(d[5]),
                    "close": _num(d[6]),
                    "spread": _num(d[7], 0.0),
                    "Trading_turnover": _int(d[8]),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # PER / PBR / 殖利率 (上市)
    # ------------------------------------------------------------------
    def taiwan_stock_per_pbr(
        self,
        stock_id: str,
        start_date: str,
        end_date: str = "",
        timeout: int = 30,
        **_,
    ) -> pd.DataFrame:
        """個股 PER/PBR TaiwanStockPER (目前僅支援上市股票)
        columns: date, stock_id, dividend_yield, PER, PBR
        """
        end_date = end_date or str(dt.date.today())
        rows = []
        for month in _month_starts(start_date, end_date):
            data = self._get_json(
                f"{TWSE}/afterTrading/BWIBBU",
                params={
                    "date": month.strftime("%Y%m%d"),
                    "stockNo": stock_id,
                    "response": "json",
                },
                timeout=timeout,
            )
            if data.get("stat") != "OK":
                continue
            for d in data.get("data", []):
                rows.append(
                    {
                        "date": _roc_to_iso(d[0]),
                        "stock_id": stock_id,
                        "dividend_yield": _num(d[1]),
                        "PER": _num(d[3]),
                        "PBR": _num(d[4]),
                    }
                )
        df = pd.DataFrame(
            rows, columns=["date", "stock_id", "dividend_yield", "PER", "PBR"]
        )
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 三大法人買賣超 (上市)
    # ------------------------------------------------------------------
    _T86_NAMES = [
        # (資料集名稱, buy 欄位 index, sell 欄位 index)
        ("Foreign_Investor", 2, 3),
        ("Foreign_Dealer_Self", 5, 6),
        ("Investment_Trust", 8, 9),
        ("Dealer_self", 12, 13),
        ("Dealer_Hedging", 15, 16),
    ]

    def taiwan_stock_institutional_investors(
        self,
        stock_id: str = "",
        start_date: str = "",
        end_date: str = "",
        timeout: int = 30,
        **_,
    ) -> pd.DataFrame:
        """個股三大法人買賣表 TaiwanStockInstitutionalInvestorsBuySell
        (目前僅支援上市;每個交易日 1 個 request)
        columns: date, stock_id, buy, name, sell
        """
        end_date = end_date or str(dt.date.today())
        rows = []
        for day in _dates(start_date, end_date):
            data = self._get_json(
                f"{TWSE}/fund/T86",
                params={
                    "date": day.strftime("%Y%m%d"),
                    "selectType": "ALLBUT0999",
                    "response": "json",
                },
                timeout=timeout,
            )
            if data.get("stat") != "OK":
                continue
            for d in data.get("data", []):
                sid = str(d[0]).strip()
                if stock_id and sid != stock_id:
                    continue
                for name, bi, si in self._T86_NAMES:
                    rows.append(
                        {
                            "date": str(day),
                            "stock_id": sid,
                            "buy": _int(d[bi]),
                            "name": name,
                            "sell": _int(d[si]),
                        }
                    )
        df = pd.DataFrame(
            rows, columns=["date", "stock_id", "buy", "name", "sell"]
        )
        return df.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 融資融券 (上市)
    # ------------------------------------------------------------------
    def taiwan_stock_margin_purchase_short_sale(
        self,
        stock_id: str = "",
        start_date: str = "",
        end_date: str = "",
        timeout: int = 30,
        **_,
    ) -> pd.DataFrame:
        """個股融資融劵表 TaiwanStockMarginPurchaseShortSale
        (目前僅支援上市;每個交易日 1 個 request)
        """
        end_date = end_date or str(dt.date.today())
        rows = []
        for day in _dates(start_date, end_date):
            data = self._get_json(
                f"{TWSE}/marginTrading/MI_MARGN",
                params={
                    "date": day.strftime("%Y%m%d"),
                    "selectType": "ALL",
                    "response": "json",
                },
                timeout=timeout,
            )
            if data.get("stat") != "OK":
                continue
            # 找出含個股明細的那張表 (fields 以「股票代號」開頭)
            detail = None
            for tbl in data.get("tables", []):
                fields = tbl.get("fields") or []
                if fields and "代號" in str(fields[0]):
                    detail = tbl
                    break
            if detail is None:
                continue
            for d in detail.get("data", []):
                sid = str(d[0]).strip()
                if stock_id and sid != stock_id:
                    continue
                rows.append(
                    {
                        "date": str(day),
                        "stock_id": sid,
                        "MarginPurchaseBuy": _int(d[2]),
                        "MarginPurchaseSell": _int(d[3]),
                        "MarginPurchaseCashRepayment": _int(d[4]),
                        "MarginPurchaseYesterdayBalance": _int(d[5]),
                        "MarginPurchaseTodayBalance": _int(d[6]),
                        "MarginPurchaseLimit": _int(d[7]),
                        "ShortSaleBuy": _int(d[8]),
                        "ShortSaleSell": _int(d[9]),
                        "ShortSaleCashRepayment": _int(d[10]),
                        "ShortSaleYesterdayBalance": _int(d[11]),
                        "ShortSaleTodayBalance": _int(d[12]),
                        "ShortSaleLimit": _int(d[13]),
                        "OffsetLoanAndShort": _int(d[14]),
                        "Note": str(d[15]).strip(),
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(["date", "stock_id"]).reset_index(drop=True)
