"""Quick integrity check of the processed panel before the full training run."""
import pandas as pd

p = pd.read_parquet("data/processed/panel.parquet")
print("rows:", len(p))
print("unique complaint_ids:", p["complaint_id"].nunique())
print("duplicate ids:", int(p["complaint_id"].duplicated().sum()))
print("null narratives:", int(p["consumer_complaint_narrative"].isna().sum()))
print("windows:", p["window_id"].nunique(), p["window_id"].min(), "->", p["window_id"].max())
sz = p.groupby("window_id").size()
print(sz.head(4))
print(sz.tail(8))
print("products:", p["product"].nunique())
