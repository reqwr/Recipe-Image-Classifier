import os
import sys
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager
from src.utils import save_keras

class ModelTraining:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_model_training_config()

    def initiate_model_training(self, train_generator, val_generator):
        logging.info("Initializing model training.")
        try:
            IMG_SIZE = (self.config.img_width, self.config.img_height)
            NUM_CLASSES = self.config.num_classes

            logging.info(f"Loading base model {self.config.base_model}.")
            base_model = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
            )
            base_model.trainable = False


            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

            model = models.Model(inputs=base_model.input, outputs=outputs)

            logging.info("Compiling model for initial training.")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate_initial),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            logging.info(f"Starting initial training for {self.config.initial_epochs} epochs.")
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.initial_epochs
            )

            logging.info(f"Fine-tuning: unfreezing top {self.config.freeze_top_layers} layers.")
            base_model.trainable = True
            for layer in base_model.layers[:-self.config.freeze_top_layers]:
                layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate_finetune),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ModelCheckpoint(self.config.model_save_path, monitor='val_accuracy', save_best_only=True)
            ]

            logging.info(f"Starting fine-tuning for {self.config.fine_tuning_epochs} epochs (top layers).")
            fine_tune_history1 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.fine_tuning_epochs,
                callbacks=callbacks
            )

            logging.info(f"Fine-tuning: freezing bottom {self.config.freeze_bottom_layers} layers.")
            for layer in base_model.layers[:self.config.freeze_bottom_layers]:
                layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate_finetune),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            logging.info(f"Starting fine-tuning for {self.config.fine_tuning_epochs} epochs (bottom layers frozen).")
            fine_tune_history2 = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config.fine_tuning_epochs,
                callbacks=callbacks
            )

            logging.info(f"Saving model.")

            save_keras(model, self.config.model_save_path)

            logging.info("Finished model training.")

            return model, history, fine_tune_history1, fine_tune_history2

        except Exception as e:
            raise CustomException(e, sys)
