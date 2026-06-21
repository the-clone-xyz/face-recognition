"""
MySQL storage for face encodings.

The application stores each registered face as one row in MySQL/MariaDB so the
data can be inspected and managed from phpMyAdmin.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mysql.connector
import numpy as np


DEFAULT_DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "facerec_user",
    "password": "facerec_password",
    "database": "facerecognition",
}


def get_db_config() -> dict[str, Any]:
    # Konfigurasi default dipakai untuk XAMPP lokal.
    # Environment variable tetap didukung agar mudah dipresentasikan di mesin lain.
    return {
        "host": os.getenv("DB_HOST", DEFAULT_DB_CONFIG["host"]),
        "port": int(os.getenv("DB_PORT", str(DEFAULT_DB_CONFIG["port"]))),
        "user": os.getenv("DB_USER", DEFAULT_DB_CONFIG["user"]),
        "password": os.getenv("DB_PASSWORD", DEFAULT_DB_CONFIG["password"]),
        "database": os.getenv("DB_NAME", DEFAULT_DB_CONFIG["database"]),
    }


class FaceDatabase:
    """Lapisan akses data untuk tabel faces.

    Setiap object FaceDatabase akan memastikan tabel tersedia, lalu memuat
    seluruh encoding dari MySQL ke memory agar pencocokan wajah bisa cepat.
    """

    def __init__(self, _legacy_path: str | None = None):
        self.config = get_db_config()
        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []
        self._ensure_schema()
        self.load()

    def _connect(self):
        # Semua query database lewat fungsi ini supaya konfigurasi hanya ada di satu tempat.
        return mysql.connector.connect(**self.config)

    def _ensure_schema(self):
        # Tabel dibuat otomatis saat aplikasi start, jadi demo tetap jalan
        # walaupun database baru saja dibuat dari phpMyAdmin/XAMPP.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS faces (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        encoding JSON NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_faces_name (name)
                    )
                    """
                )
            conn.commit()

    def load(self):
        # Encoding disimpan sebagai JSON di MySQL, lalu dikonversi kembali
        # menjadi numpy array untuk perhitungan Euclidean Distance.
        with self._connect() as conn:
            with conn.cursor(dictionary=True) as cur:
                cur.execute("SELECT name, encoding FROM faces ORDER BY id ASC")
                rows = cur.fetchall()

        self.known_names = []
        self.known_encodings = []
        for row in rows:
            encoding = row["encoding"]
            if isinstance(encoding, str):
                encoding = json.loads(encoding)
            self.known_names.append(row["name"])
            self.known_encodings.append(np.array(encoding, dtype=np.float64))

    def save(self):
        raise NotImplementedError("Use add_face/remove_face for MySQL-backed storage.")

    def add_face(self, name: str, encoding: np.ndarray):
        # Saat registrasi, satu encoding rata-rata disimpan sebagai satu record wajah.
        encoding_json = json.dumps(encoding.tolist())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO faces (name, encoding) VALUES (%s, %s)",
                    (name, encoding_json),
                )
            conn.commit()
        self.load()

    def remove_face(self, name: str) -> int:
        # Penghapusan berdasarkan nama menghapus semua sample milik nama tersebut.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM faces WHERE name = %s", (name,))
                deleted = cur.rowcount
            conn.commit()
        self.load()
        return deleted

    def unique_names(self) -> list[str]:
        return sorted(set(self.known_names))

    def count(self) -> int:
        return len(self.known_names)

    def as_dict(self) -> dict[str, list[Any]]:
        return {"encodings": self.known_encodings, "names": self.known_names}

    def rows(self) -> list[dict[str, Any]]:
        # Data ringkas untuk halaman /database; encoding penuh tidak ditampilkan
        # karena vektornya panjang, cukup dimensi dan nilai awal sebagai bukti data.
        with self._connect() as conn:
            with conn.cursor(dictionary=True) as cur:
                cur.execute(
                    """
                    SELECT
                        id,
                        name,
                        JSON_LENGTH(encoding) AS dimensions,
                        JSON_EXTRACT(encoding, '$[0]') AS first_value,
                        created_at
                    FROM faces
                    ORDER BY id DESC
                    """
                )
                return cur.fetchall()
