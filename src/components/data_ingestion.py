import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, hf_hub_url
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import ConfigurationManager

class DataIngestion:
    def __init__(self):
        config_manager = ConfigurationManager()
        self.config = config_manager.get_data_ingestion_config()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion.")
        try:
            repo_id = self.config.hf_repo_id
            subdir = self.config.hf_images_subdir
            target_dir = self.config.download_dir
            os.makedirs(target_dir, exist_ok=True)

            logging.info(f"Downloading from Hugging Face repo.")

            file_list_path = hf_hub_download(repo_id=repo_id, filename=f"{subdir}/file_list.txt")
            with open(file_list_path, "r") as f:
                files = [line.strip() for line in f.readlines()]

            for file_rel in files:
                url = hf_hub_url(repo_id=repo_id, filename=file_rel)
                dest_path = os.path.join(target_dir, Path(file_rel).name)
                # Download file
                hf_hub_download(repo_id=repo_id, filename=file_rel, local_dir=target_dir, force_download=True)
            
            logging.info(f"Finished data ingestion.")
            return target_dir

        except Exception as e:
            raise CustomException(e, sys)
