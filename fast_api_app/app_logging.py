"""Module responsible for the configuration of a given logger"""
import logging
from config import Config

def setup_logging(logger: logging.Logger):
    """Configure the logging for a specific logger."""

    # Set the log level for this specific logger
    logger.setLevel(Config.LOG_LEVEL)

    # Set up log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the level for console output

    # Create a file handler
    file_handler = logging.FileHandler("app.log", mode='a')
    file_handler.setLevel(logging.INFO)  # Set the level for file output

    # Create a formatter and attach it to the handlers
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Set log levels for external libraries
    logging.getLogger('elasticsearch').setLevel(logging.CRITICAL)
    logging.getLogger('elastic_transport.transport').setLevel(logging.CRITICAL)
    logging.getLogger('elasticsearch.trace').setLevel(logging.CRITICAL)
    logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
