from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

from src.logger import logging
from src.exception import CustomException

import sys


def main():
    try:
        logging.info("Pipeline started")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation(DataTransformationConfig())
        X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
            train_path, test_path
        )

        trainer = ModelTrainer(ModelTrainerConfig())
        score = trainer.run(X_train, y_train, X_test, y_test)

        logging.info(f"Pipeline completed | Score: {score}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()