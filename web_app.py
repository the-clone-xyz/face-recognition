from __future__ import annotations

import base64
import io
import os
import time
from collections import Counter

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

from face_db import FaceDatabase, get_db_config
from pcd_utils import (
    calculate_fps,
    convert_to_grayscale_rgb,
    get_face_recognition,
    normalize_pixels,
    recognize_face,
)


TOLERANCE = 0.55
FRAME_SCALE = 0.35
UNKNOWN_LABEL = "Tidak Dikenal"

app = Flask(__name__)


def decode_image(data_url: str) -> np.ndarray:
    # Browser mengirim frame kamera sebagai data URL base64.
    # Fungsi ini mengubahnya kembali menjadi array RGB untuk OpenCV/face_recognition.
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image)


def get_database() -> FaceDatabase:
    # FaceDatabase dibuat per request agar data terbaru dari MySQL selalu terbaca,
    # misalnya setelah registrasi atau hapus wajah.
    return FaceDatabase()


def detect_faces(rgb_image: np.ndarray):
    # PCD: grayscale memakai I(x,y)=0.299R+0.587G+0.114B.
    # Nilai normalisasi I'(x,y)=I(x,y)/255 disiapkan sebagai informasi analisis,
    # sedangkan library face_recognition tetap membutuhkan citra RGB.
    gray = convert_to_grayscale_rgb(rgb_image)
    normalized = normalize_pixels(gray)
    # Frame diperkecil sebelum deteksi agar proses lebih ringan saat kamera realtime.
    small = cv2.resize(rgb_image, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    face_recognition = get_face_recognition()
    locations = face_recognition.face_locations(small, model="hog")
    encodings = face_recognition.face_encodings(small, locations)
    scale = 1.0 / FRAME_SCALE
    # Koordinat hasil deteksi dari frame kecil dikembalikan ke ukuran asli
    # supaya kotak wajah di canvas frontend tepat posisinya.
    scaled_locations = [
        {
            "top": round(top * scale),
            "right": round(right * scale),
            "bottom": round(bottom * scale),
            "left": round(left * scale),
        }
        for top, right, bottom, left in locations
    ]
    analysis = {
        "gray_mean": round(float(np.mean(gray)), 2),
        "normalized_mean": round(float(np.mean(normalized)), 4),
    }
    return encodings, scaled_locations, analysis


@app.get("/")
def index():
    # Halaman utama: kamera, registrasi wajah, dan hasil recognition realtime.
    return render_template("index.html")


@app.get("/database")
def database_page():
    # Halaman presentasi data: menampilkan isi tabel faces dari MySQL/phpMyAdmin.
    config = get_db_config()
    try:
        db = get_database()
        rows = db.rows()
        error = None
    except Exception as exc:
        rows = []
        error = str(exc)
    return render_template("database.html", config=config, rows=rows, error=error)


@app.get("/api/status")
def status():
    # Dipanggil frontend untuk menunjukkan apakah database aktif dan berapa encoding tersimpan.
    config = get_db_config()
    try:
        db = get_database()
        return jsonify(
            {
                "ok": True,
                "database": config["database"],
                "host": config["host"],
                "count": db.count(),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "database": config["database"]}), 503


@app.get("/api/faces")
def list_faces():
    # Mengirim daftar nama unik beserta jumlah sample encoding per nama ke panel Database.
    try:
        db = get_database()
        counts = Counter(db.known_names)
        return jsonify(
            {
                "ok": True,
                "faces": [
                    {"name": name, "samples": count}
                    for name, count in sorted(counts.items())
                ],
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "faces": []}), 503


@app.delete("/api/faces/<name>")
def delete_face(name: str):
    # Tombol Hapus di frontend memanggil route ini untuk menghapus semua data wajah milik nama.
    try:
        deleted = get_database().remove_face(name)
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.post("/api/recognize")
def recognize():
    # Alur realtime:
    # frame kamera -> deteksi wajah -> encoding -> cocokkan ke database -> kirim box dan metrik.
    started_at = time.time()
    payload = request.get_json(force=True)
    rgb = decode_image(payload["image"])
    encodings, locations, analysis = detect_faces(rgb)
    db = get_database()

    results = []
    for encoding, location in zip(encodings, locations):
        name = UNKNOWN_LABEL
        metrics = recognize_face(encoding, db.known_encodings, TOLERANCE)
        if metrics.best_index is not None and metrics.status == "Dikenali":
            name = db.known_names[metrics.best_index]
        best_distance = None if metrics.distance is None else round(metrics.distance, 4)
        cosine = None if metrics.cosine_similarity is None else round(metrics.cosine_similarity, 4)
        # Keputusan utama:
        # Status = Dikenali jika d <= T, Tidak Dikenali jika d > T.
        # d dihitung dengan Euclidean Distance pada pcd_utils.recognize_face().
        results.append(
            {
                "name": name,
                "status": metrics.status,
                "confidence": metrics.confidence,
                "distance": best_distance,
                "cosine_similarity": cosine,
                "threshold": TOLERANCE,
                "box": location,
            }
        )

    process_time, fps = calculate_fps(started_at)
    return jsonify(
        {
            "ok": True,
            "faces": results,
            "registered": db.count(),
            "process_time": round(process_time, 4),
            "fps": round(fps, 2),
            "analysis": analysis,
        }
    )


@app.post("/api/register")
def register():
    # Registrasi mengambil beberapa frame, lalu menyimpan rata-rata encoding
    # agar data wajah lebih stabil dibanding satu frame saja.
    payload = request.get_json(force=True)
    name = payload.get("name", "").strip()
    images = payload.get("images", [])
    if not name:
        return jsonify({"ok": False, "error": "Nama wajib diisi."}), 400
    if not images:
        return jsonify({"ok": False, "error": "Tidak ada frame kamera."}), 400

    collected = []
    for image_data in images:
        rgb = decode_image(image_data)
        encodings, locations, _analysis = detect_faces(rgb)
        if len(locations) > 1:
            app.logger.warning("Frame registrasi memiliki lebih dari satu wajah; digunakan wajah pertama.")
        if encodings:
            collected.append(encodings[0])

    if not collected:
        return jsonify({"ok": False, "error": "Wajah tidak terdeteksi."}), 400

    # Rata-rata vektor encoding menjadi representasi final wajah yang disimpan ke MySQL.
    avg_encoding = np.mean(collected, axis=0)
    db = get_database()
    db.add_face(name, avg_encoding)
    return jsonify({"ok": True, "name": name, "samples": len(collected), "total": db.count()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
