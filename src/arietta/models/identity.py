from __future__ import annotations
import hashlib
from pathlib import Path


def repository_fingerprint(path: str) -> str:
    root = Path(path)
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}
    )
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(root)).encode())
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
