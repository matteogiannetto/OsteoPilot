# agent/logger_config.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Name of the parent logger for all your code
PARENT_LOGGER_NAME = "agent"   # you can rename, but keep it stable

def _configure_parent_logger(
    level: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """
    Configure the *parent* 'agent' logger once.
    All module loggers will be children of this one.
    """
    logger = logging.getLogger(PARENT_LOGGER_NAME)

    # Only configure once
    if getattr(logger, "_agent_configured", False):
        return logger

    # Do NOT propagate to root -> avoids polluting stdout / uvicorn logs
    logger.propagate = False

    level_name: str = level if level is not None else (os.getenv("LOG_LEVEL") or "INFO")
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # File to write *all* your custom logs
    log_file = log_file if log_file is not None else (os.getenv("LOG_FILE") or "logs/agent.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fh = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(numeric_level)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(fh)

    # Mark as configured
    logger.__dict__["_agent_configured"] = True
    return logger


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    Get a child logger under 'agent', e.g. 'agent.agent_core',
    'agent.subgraphs.imaging_preprocessing', etc.

    All of them inherit handlers from the parent and go ONLY to the log file.
    """
    # Make sure parent has handlers
    _configure_parent_logger(level=level)

    # Full child name: "agent.<module-name>"
    full_name = f"{PARENT_LOGGER_NAME}.{name}"

    logger = logging.getLogger(full_name)
    # Children should propagate up to 'agent' but not beyond
    logger.propagate = True
    # Typically no handlers on children; they use the parent's file handler
    return logger
