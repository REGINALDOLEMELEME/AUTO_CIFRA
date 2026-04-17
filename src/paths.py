from pathlib import Path


def ensure_directories(paths: dict) -> None:
    for key in ("input_dir", "output_dir", "tmp_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
