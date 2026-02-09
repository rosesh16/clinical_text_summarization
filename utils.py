import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj):
    """
    Saves any Python object using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads a pickled Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logging.error("Error occurred while loading object")
        raise CustomException(e, sys)


def create_directories(paths: list):
    """
    Creates directories if they do not exist.
    """
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logging.info(f"Directory created or exists: {path}")

    except Exception as e:
        logging.error("Error occurred while creating directories")
        raise CustomException(e, sys)