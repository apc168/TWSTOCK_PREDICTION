# -*- coding: utf-8 -*-
import sys

sys.stdout.reconfigure(encoding="utf-8")

from local_data_loader import DataLoader

dl = DataLoader()

print("=== taiwan_stock_info ===")
info = dl.taiwan_stock_info()
dl._stock_info = info
print(info.shape)
print(info[info.stock_id.isin(["2330", "6488"])])

print("\n=== taiwan_stock_daily 2330 (銝?) ===")
df = dl.taiwan_stock_daily("2330", "2026-06-01", "2026-06-11")
print(df)

print("\n=== taiwan_stock_daily 6488 (銝?) ===")
df = dl.taiwan_stock_daily("6488", "2026-06-01", "2026-06-11")
print(df)

print("\n=== taiwan_stock_per_pbr 2330 ===")
df = dl.taiwan_stock_per_pbr("2330", "2026-06-01", "2026-06-11")
print(df)

print("\n=== taiwan_stock_institutional_investors 2330 ===")
df = dl.taiwan_stock_institutional_investors(
    "2330", "2026-06-09", "2026-06-11"
)
print(df)

print("\n=== taiwan_stock_margin_purchase_short_sale 2330 ===")
df = dl.taiwan_stock_margin_purchase_short_sale(
    "2330", "2026-06-10", "2026-06-11"
)
print(df.T)

print("\nALL OK")

