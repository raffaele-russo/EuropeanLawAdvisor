"""Logger Module"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get log level from environment variable, default to INFO if not set
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)
