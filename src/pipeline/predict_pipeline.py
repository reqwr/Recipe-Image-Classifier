import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import load_keras  # optional utility if you use custom wrapper

class PredictionPipeline():
    def __init__(self, model_path, preprocessor_path):
        try:
            logging.info("Loading trained model for predictions.")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            self.model = load_keras(model_path)

            logging.info("Loading preprocessor (ImageDataGenerator).")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            with open(preprocessor_path, "rb") as f:
                self.datagen = pickle.load(f)

            self.img_width = getattr(self.datagen, "target_size", (224, 224))[0]
            self.img_height = getattr(self.datagen, "target_size", (224, 224))[1]

        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_image(self, image_path):
        try:
            img = load_img(image_path, target_size=(self.img_width, self.img_height))
            arr = img_to_array(img) / 255.0 
            arr = np.expand_dims(arr, axis=0)
            return arr
        except Exception as e:
            raise CustomException(e, sys)

    def predict_single_image(self, image_path):
        try:
            x = self.preprocess_image(image_path)
            preds = self.model.predict(x)
            return preds[0]
        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, image_paths):
        try:
            batch = np.vstack([self.preprocess_image(p) for p in image_paths])
            preds = self.model.predict(batch)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
