import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import useful as use

priv = use.get_priv()
DATA_PATH = pathlib.Path(priv["DATA_PATH"])
HOME_DIR = pathlib.Path(priv["HOME_DIR"])

## Save dictionary to json file
save_dict_to_json = use.save_dict_to_json

## Open dictionary from json file
open_dict_from_json = use.open_dict_from_json

save_model_pickle = use.save_model_pickle
load_model_pickle = use.load_model_pickle

report_progress = use.report_progress

def read_dataset_and_split(dataset_path, test_size, target_col, use_csv=True):
    if use_csv:
        df = pd.read_csv(dataset_path, index_col=0)
    else:
        df = pd.read_json(dataset_path)
    X = df.iloc[:, :target_col].values
    y = df.iloc[:, target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
    return (X_train, y_train), (X_test, y_test)

def save_split_dataset(train_set, test_set, train_path=DATA_PATH / "dataset/train_set.npz", test_path=DATA_PATH / "dataset/test_set.npz"):
    X_train, y_train = train_set
    X_test, y_test = test_set
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    np.savez(train_path, train_X=X_train, train_y=y_train)
    np.savez(test_path, test_X=X_test, test_y= y_test)
    report_progress("SAVE", train_path)
    report_progress("SAVE", test_path)

def load_train_set(train_path=DATA_PATH / "dataset/train_set.npz"):
    npz = np.load(train_path)
    X_train = npz["train_X"]
    y_train = npz["train_y"]
    report_progress("LOAD", train_path)
    return X_train, y_train

def load_test_set(test_path=DATA_PATH / "dataset/test_set.npz"):
    npz = np.load(test_path)
    X_test = npz["test_X"]
    y_test = npz["test_y"]
    report_progress("LOAD", test_path)
    return X_test, y_test


