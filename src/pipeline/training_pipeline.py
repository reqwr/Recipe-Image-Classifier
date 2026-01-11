import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

class TrainingPipeline():
    def initiate_training_pipeline(self):
        try:
            logging.info("Initiating training pipeline.")

            data_ingestion = DataIngestion()
            images_dir = data_ingestion.initiate_data_ingestion()
            data_transformation = DataTransformation()
            train_generator, val_generator, _ = data_transformation.initiate_data_transformation(download_dir=images_dir)
            model_trainer = ModelTraining()

            model_trainer.initiate_model_training(train_generator, val_generator)

            logging.info("Finished training pipeline.")
        except Exception as e:
            raise CustomException(e, sys)

