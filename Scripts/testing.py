#%%
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="../LOG/progress.log",
                    encoding="utf-8",
                    level=logging.INFO,
                    format='%(asctime)s; %(message)s')

progress_task = "SAVE"
progress_path = "Something/here/"
logger.info(f"{progress_task}; {progress_path}")

#%%
import pandas as pd

log = pd.read_csv("../LOG/progress.log", sep="; ", engine="python")
log

#%%
import useful as use
import pandas as pd
import numpy as np
from pathlib import Path
priv = use.get_priv()
API_KEY = priv["API_KEY"]
DATA_PATH = Path(priv["DATA_PATH"])