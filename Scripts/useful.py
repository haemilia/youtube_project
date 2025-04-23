############################################################################################
# Useful little functions module
############################################################################################
import pandas as pd
import json
from pathlib import Path
from datetime import date
import pickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="../LOG/progress.log",
                    encoding="utf-8",
                    level=logging.INFO,
                    format='%(asctime)s; %(message)s')

## Get private information
def get_priv(priv_path="../private.json") -> dict:
    """Retrieves private information from `private.json`.

    Args:
        priv_path: Path to `private.json`. `"../private.json"` by default.
    """
    with open(priv_path, "r", encoding="utf-8") as json_file:
        priv:dict = json.load(json_file)
    return priv

## Get today's date
def get_today()-> str:
    """Retrieve today's date in `%Y-%m-%d` format.
    """
    return str(date.today())

## Create directory if it doesn't exist
def create_directory_if_not_exists(directory_path:str|Path) -> None:
    """Creates a directory if it doesn't already exist.

    Args:
        directory_path: The path to the directory to create.
    """
    path:Path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)

## Report Progress (on CLI)
def report_progress(progress_task:str, progress_path:str|Path)->None:
    """For printing progress on CLI
    Args:
        progress_task: The task to report (capitalised by convention)
        progress_path: File path related to the task. Can be just a non-path string.
    """
    global logger
    logger.info(f"{progress_task}; {progress_path}")
    print(f"{progress_task}: {progress_path}")

## Process path file names
def get_df_paths(df_dir:Path|str)->dict:
    """Returns a dictionary of file paths. 
    - First level of keys: `collection_date`. 
    - Second level of keys: `category_id`.
    Args:
        df_dir: Path to directory that contains dataframe files.
    """
    df_dir = Path(df_dir)
    df_paths = {}
    for df_file in df_dir.iterdir():
        if not df_file.is_dir():
            collection_date, category_id= str(df_file.stem).split("_")
            # if df_paths['collection_date'] doesn't exist, make an empty dictionary as its value
            if not df_paths.get(collection_date):
                df_paths[collection_date] = {}
            df_paths[collection_date][category_id] = df_file
    return df_paths

## Save dictionary to json file
def save_dict_to_json(save_dict: dict, save_path:Path|str, report=True)-> bool:
    """Save a dictionary to a json file. Returns True if successful, returns False if not.
    Args:
        save_dict: dictionary object to save
        save_path: destination path for json file
        report: default True; reports progress on CLI if True
    """
    try:
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(save_dict, json_file, indent=4)
    except:
        return False
    if report:
        report_progress("SAVE", progress_path=save_path)
    return True

## Open dictionary from json file
def open_dict_from_json(from_path:Path|str, report=True) -> dict|bool:
    """
    Retrive dictionary from json file. If failed, returns False.
    Args:
        from_path: json file's path
        report: default True; reports progress on CLI if True 
    """
    try:
        with open(from_path, "r", encoding="utf-8") as json_file:
            json_dict = json.load(json_file)
    except:
        return False
    if report:
        report_progress("LOAD", progress_path=from_path)
    return json_dict

## Save model to pickle
def save_model_pickle(model, filepath:Path|str, report=True) -> bool:
    """
    Save ML model as pickle
    Args:
        model: ML model object
        filepath: destination path
        report: default True; reports progress on CLI if True
    """
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
    except:
        return False
    if report:
        report_progress("SAVE MODEL", filepath)
    return True

## Load model from pickle
def load_model_pickle(filepath:Path|str, report=True):
    """
    Read ML model from pickle file
    Args:
        filepath: pickle file's path
        report: default True; reports progress on CLI if True
    """
    try:
        with open(filepath, 'rb') as file:
            loaded_model = pickle.load(file)
    except:
        return False
    if report:
        report_progress("LOAD MODEL", filepath)
    return loaded_model