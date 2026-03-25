import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    input_path: str = os.path.join("data", "powerconsumption.csv")
    raw_path: str = os.path.join("artifacts", "raw.csv")
    train_path: str = os.path.join("artifacts", "train.csv")
    test_path: str = os.path.join("artifacts", "test.csv")
    test_size: float = 0.3
    random_state: int = 42


class DataIngestion:

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")

            if not os.path.isfile(self.config.input_path):
                raise FileNotFoundError("Input file not found")

            df = pd.read_csv(self.config.input_path)

            logging.info(f"Dataset shape: {df.shape}")

            os.makedirs(os.path.dirname(self.config.raw_path), exist_ok=True)

            df.to_csv(self.config.raw_path, index=False)

            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            train_df.to_csv(self.config.train_path, index=False)
            test_df.to_csv(self.config.test_path, index=False)

            return self.config.train_path, self.config.test_path

        except Exception as e:
            raise CustomException(e, sys)