import os
import sys
import pickle
import optuna
import mlflow
import dagshub
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.logger import logging
from src.exception import CustomException


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    n_trials: int = 5
    random_state: int = 1


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        try:
            dagshub.init(repo_name="EnergyConsumption_PredictionModel",
                         repo_owner="chandanc5525")

            mlflow.set_tracking_uri(
                "https://dagshub.com/chandanc5525/EnergyConsumption_PredictionModel.mlflow"
            )

            mlflow.set_experiment("Energy-ML")

            logging.info("MLflow + DagsHub initialized")

        except Exception as e:
            raise CustomException(e, sys)

        self.models = {
            "DecisionTree": DecisionTreeRegressor,
            "RandomForest": RandomForestRegressor,
            "ExtraTrees": ExtraTreesRegressor,
            "XGBoost": XGBRegressor,
            "CatBoost": CatBoostRegressor
        }

    def _objective(self, trial, model_name, X_tr, y_tr, X_val, y_val):

        if model_name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "random_state": self.config.random_state
            }

        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "random_state": self.config.random_state
            }

        elif model_name == "ExtraTrees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "random_state": self.config.random_state
            }

        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "verbosity": 0
            }

        elif model_name == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "depth": trial.suggest_int("depth", 3, 10),
                "verbose": 0
            }

        model = self.models[model_name](**params)

        # Model Fitting
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        return r2_score(y_val, preds)

    def run(self, X_train, y_train, X_test, y_test):

        try:
            logging.info("Model training started")

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=self.config.random_state
            )

            best_model = None
            best_score = -np.inf
            best_model_name = None

            for name in self.models.keys():

                logging.info(f"Training model: {name}")

                # Start with Optuna Study 
                study = optuna.create_study(
                    study_name=f"{name}_study",
                    direction="maximize"
                )

                study.optimize(
                    lambda trial: self._objective(
                        trial, name, X_tr, y_tr, X_val, y_val
                    ),
                    n_trials=self.config.n_trials
                )

                model = self.models[name](**study.best_params)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                score = r2_score(y_test, preds)

                with mlflow.start_run(run_name=name):
                    mlflow.log_params(study.best_params)
                    mlflow.log_metric("r2_score", score)

                logging.info(f"{name} score: {score}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name

            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

            with open(self.config.model_path, "wb") as f:
                pickle.dump(best_model, f)

            logging.info(f"Best Model: {best_model_name} | Score: {best_score}")

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)