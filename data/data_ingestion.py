import pandas as pd


def data_ingestion(path):
    return pd.read_csv(path)