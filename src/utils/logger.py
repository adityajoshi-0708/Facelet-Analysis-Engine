import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DEFAULT_LEVEL = logging.INFO


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level or _DEFAULT_LEVEL)

    # Console handler
    ch = logging.StreamHandler()
    try:
        if hasattr(ch.stream, "reconfigure"):
            ch.stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    ch.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(fh)

    logger.propagate = False
    return logger