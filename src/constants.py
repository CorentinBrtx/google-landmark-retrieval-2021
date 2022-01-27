import logging
import os


class Constants:
    """Constants."""

    LOGGING_LEVEL = logging.INFO

    SRC_FOLDER = os.path.dirname(os.path.abspath(__file__))
    ROOT_FOLDER = os.path.dirname(SRC_FOLDER)
