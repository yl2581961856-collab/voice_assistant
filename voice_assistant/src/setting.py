"""Configuration helpers for the voice assistant stack."""
from __future__ import annotations

import logging.config
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("`pyyaml` is required for configuration loading. Install with `pip install pyyaml`.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_LOGGING_PATH = PROJECT_ROOT / "config" / "logging.conf"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return data


def configure_logging(config_path: Optional[str | Path] = None) -> None:
    log_path = Path(config_path) if config_path else DEFAULT_LOGGING_PATH
    if log_path.exists():
        logging.config.fileConfig(log_path, disable_existing_loggers=False)
    else:
        logging.basicConfig(level=logging.INFO)
