from dataclasses import dataclass
import yaml

@dataclass
class DataIngestionConfig:
    hf_repo_id: str
    hf_images_subdir: str
    download_dir: str

@dataclass
class DataTransformationConfig:
    img_width: int
    img_height: int
    batch_size: int
    test_size: float
    preprocessor_save_path: str


@dataclass
class ModelTrainingConfig:
    img_width: int
    img_height: int
    num_classes: int
    base_model: str
    initial_epochs: int
    fine_tuning_epochs: int
    learning_rate_initial: float
    learning_rate_finetune: float
    freeze_top_layers: int
    freeze_bottom_layers: int
    model_save_path: str

class ConfigurationManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            hf_repo_id=self.config["data_ingestion"]["hf_repo_id"],
            hf_images_subdir=self.config["data_ingestion"]["hf_images_subdir"],
            download_dir=self.config["data_ingestion"]["download_dir"]
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            img_width=self.config["data_transformation"]["img_width"],
            img_height=self.config["data_transformation"]["img_height"],
            batch_size=self.config["data_transformation"]["batch_size"],
            test_size=self.config["data_transformation"]["test_size"],
            preprocessor_save_path=self.config["data_transformation"]["preprocessor_save_path"]
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        return ModelTrainingConfig(
            img_width=self.config["model_training"]["img_width"],
            img_height=self.config["model_training"]["img_height"],
            num_classes=self.config["model_training"]["num_classes"],
            base_model=self.config["model_training"]["base_model"],
            initial_epochs=self.config["model_training"]["initial_epochs"],
            fine_tuning_epochs=self.config["model_training"]["fine_tuning_epochs"],
            learning_rate_initial=self.config["model_training"]["learning_rate_initial"],
            learning_rate_finetune=self.config["model_training"]["learning_rate_finetune"],
            freeze_top_layers=self.config["model_training"]["freeze_top_layers"],
            save_path=self.config["model_training"]["save_path"],
            freeze_top_layers=self.config["model_training"]["freeze_top_layers"],
            freeze_bottom_layers=self.config["model_training"]["freeze_bottom_layers"],
            model_save_path=self.config["model_training"]["model_save_path"]
        )
