# TWSTOCK

TWSTOCK 是一個本機版的台股研究工具。

它可以幫你自動整理台股日線資料、建立技術指標、訓練 LightGBM 模型，再套用重大新聞 / 重大訊息 / 下市風險過濾，最後輸出可供人工檢查的股票雷達與手機可看的 HTML 報表。

> 重要提醒：本專案只適合用於學習、研究與技術實驗，不是投資建議。模型預測可能錯誤，歷史回測不代表未來績效。

---

## ETF 說明

本專案**不包含 ETF**。

TWSTOCK 的研究範圍聚焦在台灣上市、上櫃的一般股票，不把 ETF 納入模型訓練、預測清單或回測結果。

這樣做的原因是：

- ETF 的價格行為和一般個股不同
- ETF 常受成分股、追蹤指數、折溢價與再平衡影響
- 如果把 ETF 和個股混在一起訓練，模型解讀會比較混亂
- 本專案目標是建立「台股個股觀察清單」，不是 ETF 篩選器

---

## 這個專案可以做什麼？

TWSTOCK 可以簡單理解成：

> 台股資料收集 + 技術指標 + 5 日方向預測 + 重大事件風險過濾 + 股票雷達 + HTML 報表

它主要可以完成以下工作：

1. 建立上市、上櫃一般股票清單
2. 抓取每檔股票的歷史日線資料
3. 將每檔股票存成獨立 CSV
4. 合併所有股票資料成 `prices/all_price.csv`
5. 建立技術指標，例如報酬率、均線乖離、RSI、成交量變化等
6. 使用 LightGBM 模型預測未來 5 個交易日方向
7. 套用重大新聞、重大訊息、下市風險等人工風險清單
8. 產生 `final_stock_radar.csv` 作為最終股票雷達
9. 產生回測報告與特徵重要性報告
10. 將 CSV 報表轉成手機可看的 HTML 頁面

---

## 適合誰使用？

這個專案適合：

- 想學 Python 股票資料處理的人
- 想做台股量化研究的人
- 想建立自己的台股觀察清單的人
- 想練習 LightGBM、特徵工程、回測流程的人
- 想每天收盤後自動產生股票雷達報表的人

不適合：

- 想要保證賺錢的人
- 想直接拿模型訊號下單的人
- 想做即時盤中交易的人
- 不理解回測風險與資料品質問題的人

---

## 專案結構

```text
TWSTOCK/
├── prices/
│   ├── 1101_price.csv
│   ├── 1102_price.csv
│   ├── ...
│   └── all_price.csv
├── html/
│   ├── index.html
│   ├── final_stock_radar.html
│   ├── prediction_5d_lightgbm.html
│   ├── backtest_report.html
│   ├── backtest_daily_topN.html
│   └── feature_importance_lightgbm.html
├── run_all.py
├── run_daily.py
├── update_prices_incremental.py
├── make_stocks_txt.py
├── example.py
├── local_data_loader.py
├── train_lightgbm_5d.py
├── apply_risk_filter.py
├── risk_watchlist.csv
├── csv_to_html.py
├── predict_tomorrow.py
└── README.md
```

說明：

| 路徑 / 檔案 | 用途 |
|---|---|
| `prices/` | 存放每檔股票的歷史價格 CSV |
| `prices/all_price.csv` | 所有股票合併後的資料 |
| `html/` | 產生給手機或瀏覽器看的 HTML 報表 |
| `run_all.py` | 第一次完整建立資料與報表 |
| `run_daily.py` | 每天收盤後增量更新 |
| `update_prices_incremental.py` | 只補缺少的日期資料 |
| `make_stocks_txt.py` | 產生台股一般股票清單，不包含 ETF |
| `example.py` | 批次產生多檔股票 CSV 的範例 |
| `local_data_loader.py` | 本地資料載入與抓取工具 |
| `train_lightgbm_5d.py` | 訓練 LightGBM 模型並產生預測與回測 |
| `apply_risk_filter.py` | 將重大事件風險清單套用到模型預測，產生最終股票雷達 |
| `risk_watchlist.csv` | 人工維護的重大新聞 / 重大訊息 / 下市風險清單 |
| `csv_to_html.py` | 將 CSV 報表轉成 HTML |
| `predict_tomorrow.py` | 規則式的簡單預測範例 |

---

## 安裝需求

建議使用 Python 3.10 以上。

```bash
pip install pandas numpy requests lxml html5lib lightgbm
```

如果你使用 Anaconda，也可以用：

```bash
conda install -c conda-forge lightgbm
```

---

## 第一次使用：完整建立資料

第一次使用時，需要先建立股票清單與歷史資料。

### 1. 產生股票清單

```powershell
python make_stocks_txt.py
```

會產生：

```text
stocks.txt
stocks_meta.csv
```

產生的股票清單只包含台灣上市、上櫃一般股票，**不包含 ETF**。

### 2. 建立價格資料夾

```powershell
mkdir prices
```

### 3. 執行完整流程

```powershell
python run_all.py --start 2023-01-01 --csv-dir prices --output-dir . --sleep 1 --retries 3 --retry-sleep 5 --top-n 50 --resume
```

這個流程會做：

1. 產生或更新股票清單
2. 抓取歷史日線資料
3. 將每檔股票存到 `prices/`
4. 建立 `prices/all_price.csv`
5. 訓練 LightGBM 模型
6. 輸出預測與回測報告
7. 產生 HTML 報表

第一次跑會花比較久，因為要一檔一檔抓歷史資料。

如果中途被中斷，可以用同一個指令重新執行，並保留 `--resume`，已經抓過的資料會跳過。

---

## 系統流程

目前流程如下：

```text
價格資料 / 技術指標
        ↓
LightGBM 5 日方向預測
        ↓
prediction_5d_lightgbm.csv
        ↓
重大新聞 / 重大訊息 / 下市風險過濾
        ↓
final_stock_radar.csv
        ↓
HTML 報表
```

LightGBM 只負責根據價格、成交量與技術指標做排序。

`apply_risk_filter.py` 會再讀取 `risk_watchlist.csv`，把有重大事件風險的股票標示為：

| final_signal | 說明 |
|---|---|
| `WATCH` | 可列入觀察 |
| `WATCH_WITH_CAUTION` | 有中度風險，僅能謹慎觀察 |
| `BLOCKED` | 有高度風險，不採用模型訊號 |

這樣可以避免模型只因為技術指標看起來超跌，就把有重大風險的股票放到觀察清單前面。

---

## 重大風險清單

風險清單檔案是：

```text
risk_watchlist.csv
```

格式如下：

```csv
stock_id,risk_level,reason,source,event_date,expire_date
6806,HIGH,使用者手動標註重大事件風險，暫不納入模型觀察清單,manual,2026-06-16,2026-12-31
```

欄位說明：

| 欄位 | 說明 |
|---|---|
| `stock_id` | 股票代號 |
| `risk_level` | `LOW` / `MEDIUM` / `HIGH` |
| `reason` | 風險原因 |
| `source` | 來源，例如 `manual`、新聞、重大訊息 |
| `event_date` | 事件日期 |
| `expire_date` | 風險標記到期日，過期後自動忽略 |

如果某檔股票被標記為 `HIGH`，最後輸出的 `final_stock_radar.csv` 會把它標成 `BLOCKED`。

---

## 每天收盤後更新

第一次完整建立資料後，之後不用每天重抓全部歷史資料。

每天收盤後執行：

```powershell
python run_daily.py --csv-dir prices --output-dir .
```

這個指令會：

1. 檢查每檔股票缺少哪些日期
2. 只補新的日線資料
3. 重建 `prices/all_price.csv`
4. 重新訓練模型
5. 套用 `risk_watchlist.csv` 產生 `final_stock_radar.csv`
6. 更新預測、風險過濾與回測報告
7. 重新產生 HTML 報表

建議在台股收盤後、日線資料比較穩定時再執行。

---

## 常用指令

### 測試前 20 檔，不跑完整流程

```powershell
python run_daily.py --csv-dir prices --output-dir . --max-stocks 20 --skip-train --skip-html
```

### 只更新資料，不重新訓練模型

```powershell
python run_daily.py --csv-dir prices --output-dir . --skip-train
```

### 更新資料與模型，但不產生 HTML

```powershell
python run_daily.py --csv-dir prices --output-dir . --skip-html
```

### 只重新套用風險過濾

如果你只改了 `risk_watchlist.csv`，可以直接跑：

```powershell
python apply_risk_filter.py --prediction prediction_5d_lightgbm.csv --risk-watchlist risk_watchlist.csv --output final_stock_radar.csv
```

### 如果有缺少個股 CSV，允許自動回補

```powershell
python run_daily.py --csv-dir prices --output-dir . --backfill-missing
```

### 資料來源不穩時，用保守參數

```powershell
python run_daily.py --csv-dir prices --output-dir . --sleep 2 --retries 6 --retry-sleep 15
```

### 查看目前有幾檔股票 CSV

```powershell
(Get-ChildItem .\prices -Filter "*_price.csv" | Where-Object { $_.Name -match '^[0-9A-Za-z]+_price\.csv$' }).Count
```

---

## 模型在預測什麼？

模型預測的是：

> 某檔股票 5 個交易日後的收盤價，是否高於今天的收盤價。

概念上等於：

```python
future_return_5d = close.shift(-5) / close - 1
target_5d_up = future_return_5d > 0
```

也就是：

- 如果 5 個交易日後上漲，標記為 1
- 如果 5 個交易日後沒有上漲，標記為 0

模型輸出的重點不是「保證哪一檔會漲」，而是幫你做排序，找出比較值得觀察的股票清單。

---

## 主要輸出檔案

| 檔案 | 說明 |
|---|---|
| `final_stock_radar.csv` | 套用重大風險過濾後的最終股票雷達 |
| `blocked_stock_radar.csv` | 被高風險規則擋下的股票清單 |
| `prediction_5d_lightgbm.csv` | LightGBM 原始 5 日方向預測 |
| `backtest_report.csv` | train / valid / test 的分類評估結果 |
| `backtest_daily_topN.csv` | 測試期間每日模型挑出的 Top-N 股票 |
| `backtest_topN_summary.json` | Top-N 回測摘要 |
| `feature_importance_lightgbm.csv` | LightGBM 特徵重要性 |
| `daily_update_report.csv` | 每日更新狀態 |
| `daily_pipeline_log.txt` | 每日流程 log |

---

## HTML 報表

CSV 報表可以轉成 HTML，方便用手機或瀏覽器查看。

主要入口：

```text
html/index.html
```

常見 HTML 報表：

| HTML | 說明 |
|---|---|
| `html/final_stock_radar.html` | 套用重大風險過濾後的最終股票雷達 |
| `html/blocked_stock_radar.html` | 被高風險規則擋下的股票清單 |
| `html/prediction_5d_lightgbm.html` | LightGBM 原始 5 日方向預測 |
| `html/backtest_report.html` | 模型 train / valid / test 評估 |
| `html/backtest_daily_topN.html` | 每日 Top-N 回測結果 |
| `html/feature_importance_lightgbm.html` | 特徵重要性 |

手動轉 HTML：

```powershell
python csv_to_html.py --csv-dir . --output-dir html --include "prediction*.csv" "backtest*.csv" "feature_importance*.csv"
```

---

## 如何看報表？

### `final_stock_radar.csv`

這份是最終股票雷達，建議優先看這份。

它是在 `prediction_5d_lightgbm.csv` 的基礎上，再套用 `risk_watchlist.csv` 產生。

重要欄位：

| 欄位 | 說明 |
|---|---|
| `final_rank` | 風險過濾後的排序 |
| `final_signal` | `WATCH` / `WATCH_WITH_CAUTION` / `BLOCKED` |
| `news_risk_level` | `LOW` / `MEDIUM` / `HIGH` |
| `news_risk_reason` | 重大事件或風險原因 |
| `news_event_date` | 事件日期 |
| `news_source` | 風險來源 |
| `is_blocked` | 是否被高風險規則擋下 |

如果 `final_signal` 是 `BLOCKED`，代表即使模型看漲，也不採用該模型訊號。

### `prediction_5d_lightgbm.csv`

這份是 LightGBM 的原始預測結果，尚未套用重大風險過濾。

重要欄位：

| 欄位 | 說明 |
|---|---|
| `stock_id` | 股票代號 |
| `date` | 資料日期 |
| `close` | 收盤價 |
| `prediction_5d` | 模型判斷 5 日後方向 |
| `prob_up_5d` | 模型判斷上漲的機率 |
| `prob_down_5d` | 模型判斷下跌的機率 |
| `rsi_14` | 14 日 RSI |
| `volume_ratio_5` | 當前成交量相對近 5 日均量 |

通常可以先看 `prob_up_5d` 排名前面的股票，再搭配成交量、RSI、均線乖離等欄位判斷是否過熱或流動性不足。

### `backtest_report.csv`

這份是分類模型的基本評估，例如：

- accuracy
- precision
- recall
- f1
- auc

注意：股票選股不應只看 accuracy。  
更重要的是模型挑出的 Top-N 股票，未來平均報酬是否穩定優於市場或簡單策略。

### `backtest_daily_topN.csv`

這份比較接近實際選股用途。

它會記錄每天模型挑出的 Top-N 股票，以及這些股票未來 5 日的平均報酬與命中率。

---

## 建議使用方式

比較好的使用方式是：

1. 每天收盤後更新資料
2. 優先查看最新 `final_stock_radar.csv`
3. 如果某檔股票有重大新聞、重大訊息或下市風險，加入 `risk_watchlist.csv`
4. 只把模型結果當成觀察清單
5. 搭配基本面、籌碼、成交量、產業題材與大盤趨勢
6. 定期檢查回測與風險清單是否仍然有效

不建議：

- 只看 `prob_up_5d` 就直接買
- 忽略 `final_signal = BLOCKED` 的股票
- 忽略交易成本
- 忽略流動性
- 用盤中資料硬跑日線模型
- 看到短期回測好就認為未來一定有效

---

## 注意事項

1. 本專案使用日線資料，不是盤中即時交易系統。
2. 本專案不包含 ETF，僅針對台灣上市、上櫃一般股票。
3. 建議收盤後再執行每日更新。
4. 資料來源可能不穩，若更新失敗可以增加 `--sleep`、`--retries`、`--retry-sleep`。
5. 模型預測不代表一定會發生。
6. 重大風險清單需要人工維護，系統不會自動保證所有新聞都被捕捉。
7. 回測績效不代表未來績效。
8. 實際交易還需要考慮手續費、證交稅、滑價、流動性與風險控管。

---

## 免責聲明

本專案僅供教育、研究與技術實驗使用。

所有模型輸出、預測結果、回測結果，都不構成任何投資建議、買賣建議或報酬保證。

使用者應自行判斷風險，並對自己的投資決策負責。
