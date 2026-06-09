"""
Import old local face databases into MySQL.

Supports the previous pickle format used by cli_mode.py and the JSON format
used by older main.py versions.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from face_db import FaceDatabase


def load_source(path: Path) -> dict:
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def migrate(path: Path):
    data = load_source(path)
    names = data.get("names", [])
    encodings = data.get("encodings", [])
    if len(names) != len(encodings):
        raise ValueError("Jumlah names dan encodings tidak sama.")

    db = FaceDatabase()
    for name, encoding in zip(names, encodings):
        db.add_face(name, np.array(encoding, dtype=np.float64))
    print(f"Imported {len(names)} face encoding(s) from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import local face database into MySQL.")
    parser.add_argument("path", nargs="?", default="face_database.pkl")
    args = parser.parse_args()
    migrate(Path(args.path))
