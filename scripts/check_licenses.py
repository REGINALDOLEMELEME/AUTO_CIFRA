"""S-6: Confirm that every installed dependency is under a free license."""
from __future__ import annotations

import importlib.metadata as md
import sys

ALLOWED_KEYWORDS = (
    "mit",
    "apache",
    "bsd",
    "0bsd",
    "isc",
    "psf",
    "python software foundation",
    "lgpl",
    "gpl",
    "mpl",
    "mozilla",
    "zlib",
    "unlicense",
    "public domain",
    "cc0",
)


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def main() -> int:
    ok = 0
    suspicious: list[tuple[str, str]] = []
    for dist in md.distributions():
        name = dist.metadata["Name"] or "unknown"
        lic = normalize(dist.metadata.get("License", ""))
        classifiers = [
            c
            for c in (dist.metadata.get_all("Classifier") or [])
            if "License" in c
        ]
        classifier_text = " ".join(classifiers).lower()
        free = any(kw in lic for kw in ALLOWED_KEYWORDS) or any(
            kw in classifier_text for kw in ALLOWED_KEYWORDS
        )
        if free:
            ok += 1
        else:
            suspicious.append((name, lic or classifier_text or "(no license metadata)"))

    print(f"Free-license packages: {ok}")
    if suspicious:
        print(f"\nPackages to review manually ({len(suspicious)}):")
        for n, l in sorted(suspicious):
            print(f"  - {n:<40} {l}")
        return 1
    print("All packages passed the free-license check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
