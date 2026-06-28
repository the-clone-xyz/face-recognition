"""
Utilitas Pengolahan Citra Digital untuk sistem face recognition.

Modul ini sengaja dibuat sederhana agar rumus yang dipakai pada makalah/tugas
dapat dilihat langsung hubungannya dengan kode program.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_face_recognition = None


def get_face_recognition():
    # Import face_recognition dibuat lazy agar aplikasi lebih cepat start.
    # Library baru dimuat saat deteksi/registrasi pertama kali digunakan.
    global _face_recognition
    if _face_recognition is None:
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\..*",
            category=UserWarning,
            module=r"face_recognition_models",
        )
        import face_recognition

        _face_recognition = face_recognition
    return _face_recognition


@dataclass
class RecognitionMetrics:
    distance: float | None
    cosine_similarity: float | None
    threshold: float
    status: str
    confidence: float
    best_index: int | None


def convert_to_grayscale_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Konversi BGR ke grayscale.

    Rumus PCD:
        I(x,y) = 0.299R + 0.587G + 0.114B

    OpenCV menerima frame kamera dalam urutan BGR, lalu menerapkan bobot
    luminance tersebut untuk menghasilkan citra 1 kanal.
    """
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def convert_to_grayscale_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """Konversi RGB ke grayscale dengan rumus luminance PCD."""
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)


def normalize_pixels(gray_image: np.ndarray) -> np.ndarray:
    """Normalisasi piksel grayscale ke rentang 0..1.

    Rumus PCD:
        I'(x,y) = I(x,y) / 255
    """
    return gray_image.astype(np.float32) / 255.0


def calculate_euclidean_distance(current_encoding: np.ndarray, known_encoding: np.ndarray) -> float:
    """Hitung jarak Euclidean antar vektor fitur wajah.

    Rumus:
        d = sqrt(sum((x_i - y_i)^2))
    """
    return float(np.linalg.norm(current_encoding - known_encoding))


def calculate_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Hitung cosine similarity antar vektor fitur wajah.

    Rumus:`
        cos(theta) = (A . B) / (|A| |B|)
    """
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def recognize_face(
    current_encoding: np.ndarray,
    known_encodings: Iterable[np.ndarray],
    threshold: float,
) -> RecognitionMetrics:
    """Cari wajah paling mirip menggunakan Euclidean Distance sebagai keputusan utama."""
    # Alur presentasi:
    # 1. Hitung jarak encoding kamera ke semua encoding database.
    # 2. Ambil jarak terkecil sebagai kandidat identitas.
    # 3. Bandingkan jarak tersebut dengan threshold untuk keputusan akhir.
    known_list = list(known_encodings)
    if not known_list:
        return RecognitionMetrics(None, None, threshold, "Database Kosong", 0.0, None)

    distances = np.array(
        [calculate_euclidean_distance(current_encoding, known_encoding) for known_encoding in known_list],
        dtype=np.float64,
    )
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    cosine = calculate_cosine_similarity(current_encoding, known_list[best_index])
    status = "Dikenali" if best_distance <= threshold else "Tidak Dikenali"
    confidence = max(0.0, min(100.0, (1.0 - best_distance) * 100.0))
    return RecognitionMetrics(best_distance, cosine, threshold, status, round(confidence, 1), best_index)


def calculate_fps(start_time: float, end_time: float | None = None) -> tuple[float, float]:
    """Hitung waktu proses frame dan FPS.

    Rumus:
        T_proses = t_akhir - t_awal
        FPS = 1 / T_proses
    """
    finish = time.time() if end_time is None else end_time
    process_time = max(0.000001, finish - start_time)
    return process_time, 1.0 / process_time


def evaluate_classification(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    """Fungsi evaluasi opsional saat data uji sudah tersedia."""
    # Fungsi ini tidak wajib untuk runtime, tetapi berguna saat presentasi evaluasi
    # akurasi jika sudah ada data uji berlabel.
    total = tp + tn + fp + fn
    accuracy = ((tp + tn) / total * 100.0) if total else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def load_known_faces_from_dataset(dataset_dir: str) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Loader dataset opsional untuk folder gambar.

    Struktur sederhana yang didukung:
        dataset/NamaOrang/gambar1.jpg
        dataset/NamaOrang/gambar2.png

    Validasi yang dilakukan:
    - folder tidak ditemukan
    - dataset kosong
    - gambar tidak terbaca
    - gambar tanpa wajah
    - gambar berisi lebih dari satu wajah
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path dataset bukan folder: {dataset_dir}")

    image_paths = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise ValueError(f"Dataset kosong: tidak ada file gambar di {dataset_dir}")

    encodings: list[np.ndarray] = []
    names: list[str] = []
    warnings: list[str] = []
    face_recognition = get_face_recognition()

    for image_path in sorted(image_paths):
        # Dataset folder dibaca satu per satu, lalu nama orang diambil dari
        # nama folder parent agar struktur dataset mudah dijelaskan.
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            warnings.append(f"Gambar tidak terbaca, dilewati: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(image_rgb, model="hog")
        if not locations:
            warnings.append(f"Tidak ada wajah, dilewati: {image_path}")
            continue
        if len(locations) > 1:
            warnings.append(f"Lebih dari satu wajah, dipakai wajah pertama: {image_path}")

        face_encodings = face_recognition.face_encodings(image_rgb, [locations[0]])
        if not face_encodings:
            warnings.append(f"Encoding gagal dibuat, dilewati: {image_path}")
            continue

        encodings.append(face_encodings[0])
        names.append(image_path.parent.name)

    if not encodings:
        raise ValueError("Dataset ditemukan, tetapi tidak ada encoding wajah yang berhasil dibuat.")

    return encodings, names, warnings
