from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
