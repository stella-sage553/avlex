"""A single tiny logging helper so modules share one configured logger."""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def get_logger(name: str = "avlex") -> logging.Logger:
    """Return a namespaced logger, attaching a stream handler once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    return logger
