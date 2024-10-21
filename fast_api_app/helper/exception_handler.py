"""Exception Handler Module"""
import functools
import logging
from fastapi import HTTPException
from .app_logging import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(logger)

class RagException(Exception):
    """Base class for exceptions in the RAG module."""
    def __init__(self, message: str) -> None:
        super().__init__(message)

class SearchException(Exception):
    """Base class for exceptions in the Search module."""
    def __init__(self, message: str) -> None:
        super().__init__(message)

def exception_handler(error_message="An error occurred"):
    """
    Decorator to log errors and handle exceptions flexibly with a custom handler.

    Args:
        error_message (str): The default error message to log and include in the exception.
        exception_handler_func (callable): Optional custom exception handler. 
        If None, default behavior is to call default_exception_handler.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (SearchException, RagException) as e:
                logger.error("Error message: %s Exception: %s", error_message, e)
                raise
            except Exception as e:
                logger.error("Unexpected Exception: %s", e)
                raise HTTPException(status_code=500,
                detail="Something went wrong. Please retry later.") from e

        return wrapper
    return decorator
