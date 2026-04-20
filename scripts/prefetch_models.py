"""Pre-download the heavy ML models into AUTO_CIFRA/models/ so first pipeline
run is fast and the installation can be verified offline afterwards (S-7)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.models import get_demucs, get_whisper, get_wav2vec_aligner, release_all


def main() -> int:
    settings = get_settings()
    print(f"Models cache: {settings.models_dir}")

    print(f"Whisper ({settings.asr.model}, int8)...")
    _, err = get_whisper(
        model_size=settings.asr.model,
        device=settings.asr.device,
        compute_type=settings.asr.compute_type,
    )
    if err:
        print(f"  FAIL: {err}")
    else:
        print("  OK")

    print(f"Demucs ({settings.separation.model})...")
    _, err = get_demucs(settings.separation.model)
    if err:
        print(f"  FAIL: {err}")
    else:
        print("  OK")

    print(f"wav2vec2 alignment ({settings.alignment.model})...")
    _, _, err = get_wav2vec_aligner(
        language=settings.asr.language,
        model_name=settings.alignment.model,
    )
    if err:
        print(f"  FAIL: {err}")
    else:
        print("  OK")

    release_all()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
