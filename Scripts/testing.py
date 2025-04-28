#%%
import useful as use
import pandas as pd
from pathlib import Path

priv = use.get_priv()
DATA_PATH = Path(priv["DATA_PATH"])

popular_df_raw = pd.read_pickle(DATA_PATH / "temp/raw/popular_raw.pkl")
# unpopular_df_raw = pd.read_pickle(DATA_PATH/"temp/raw/unpopular_raw.pkl")
row = popular_df_raw[popular_df_raw["id"] == "7nGdpHQ0Uc8"]
row["title"].iloc[0]