import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _ROOT / "configs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_name: str = "config.yaml") -> Dict[str, Any]:
    """Load main config and merge all sub-configs."""
    main_path = _CONFIGS_DIR / config_name
    if not main_path.exists():
        raise FileNotFoundError(f"Config not found: {main_path}")

    cfg = _load_yaml(main_path)

    # Merge sub-configs
    for sub in ["retrieval.yaml", "scoring.yaml", "models.yaml", "logging.yaml"]:
        sub_path = _CONFIGS_DIR / sub
        if sub_path.exists():
            key = sub.replace(".yaml", "")
            cfg.setdefault(key, {}).update(_load_yaml(sub_path))

    return cfg


def get_section(section: str, config_name: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_name)
    return cfg.get(section, {})