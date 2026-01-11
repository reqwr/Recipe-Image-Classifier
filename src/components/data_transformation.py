import os
import sys
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager

class DataTransformation:
    def __init__(self, config):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_transformation_config()

    def initiate_data_transformation(self, download_dir):
        logging.info("Initiating data transformation.")
        try:
            logging.info("Creating preprocessors.")

            datagen = ImageDataGenerator(
                rescale=1.0/255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                validation_split=self.config.test_size
            )

            train_generator = datagen.flow_from_directory(
                directory=download_dir,
                target_size=(self.config.img_width, self.config.img_height),
                batch_size=self.config.batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )

            val_generator = datagen.flow_from_directory(
                directory=download_dir,
                target_size=(self.config.img_width, self.config.img_height),
                batch_size=self.config.batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )

            logging.info("Saving preprocessor.")

            os.makedirs(self.config.preprocessor_save_path, exist_ok=True)
            preprocessor_file = os.path.join(self.config.preprocessor_save_path, "datagen.pkl")
            with open(preprocessor_file, "wb") as f:
                pickle.dump(datagen, f)

            logging.info(f"Finished data transformation.")

            return train_generator, val_generator, preprocessor_file

        except Exception as e:
            raise CustomException(e, sys)
