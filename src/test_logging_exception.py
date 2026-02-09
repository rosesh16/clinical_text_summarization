import sys
from logger import logging
from exception import CustomException

logging.info("Testing CustomException with logging")

try:
    x = int("abc")
except Exception as e:
    logging.error("Type conversion failed")
    raise CustomException(e, sys)