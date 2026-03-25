import os
import sys
import pickle
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    target_column: str = "PowerConsumption_Zone3"
    preprocessor_path: str = os.path.join(
        "artifacts", "data_transformation", "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_preprocessor(self, X: pd.DataFrame):

        num_cols = X.select_dtypes(exclude="object").columns.tolist()
        cat_cols = X.select_dtypes(include="object").columns.tolist()

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Data transformation started")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target = self.config.target_column

            if target not in train_df.columns:
                raise ValueError(f"Target column '{target}' not found")

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            preprocessor = self.get_preprocessor(X_train)

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)

            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)