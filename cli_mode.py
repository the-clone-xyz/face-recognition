"""
Face Recognition - CLI Mode (tanpa GUI)
========================================
Jalankan ini untuk test basic face recognition via terminal.
Berguna untuk troubleshooting sebelum pakai GUI.

Usage:
    python cli_mode.py --mode recognize
    python cli_mode.py --mode register --name "Nama Anda"
"""

import cv2
import face_recognition
import numpy as np
import argparse
import time
from face_db import FaceDatabase
from pcd_utils import (
    calculate_fps,
    convert_to_grayscale_bgr,
    normalize_pixels,
    recognize_face,
    evaluate_classification,
    load_known_faces_from_dataset,
)

TOLERANCE = 0.6
SCALE     = 0.5


def load_db():
    return FaceDatabase().as_dict()


def register_face(name: str, n_samples: int = 5):
    """Ambil n_samples encoding wajah dan simpan ke database."""
    db  = FaceDatabase()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera tidak ditemukan!")
        return

    print(f"\n📸 Mendaftarkan wajah: {name}")
    print(f"   Posisikan wajah Anda di depan kamera. Butuh {n_samples} sample.")
    print("   Tekan 'q' untuk batal.\n")

    encodings  = []
    last_time  = 0

    while len(encodings) < n_samples:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame kamera gagal dibaca.")
            time.sleep(0.05)
            continue

        frame     = cv2.flip(frame, 1)
        # PCD: grayscale menerapkan I(x,y)=0.299R+0.587G+0.114B.
        gray      = convert_to_grayscale_bgr(frame)
        # PCD: normalisasi menerapkan I'(x,y)=I(x,y)/255.
        normalized = normalize_pixels(gray)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 255), 2)

        info = f"Sample: {len(encodings)}/{n_samples}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        process_time, fps = calculate_fps(start_time)
        cv2.putText(frame, f"FPS: {fps:.2f} | Process: {process_time:.4f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(frame, f"Gray mean: {np.mean(gray):.2f} | Norm mean: {np.mean(normalized):.4f}", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)
        cv2.imshow(f"Register: {name}", frame)

        now = time.time()
        if locations and (now - last_time) >= 0.6:
            enc_list = face_recognition.face_encodings(rgb, locations)
            if enc_list:
                encodings.append(enc_list[0])
                last_time = now
                print(f"   ✅ Sample {len(encodings)}/{n_samples} diambil")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("   ⚠ Registrasi dibatalkan.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(encodings) == n_samples:
        avg = np.mean(encodings, axis=0)
        db.add_face(name, avg)
        print(f"\n✅ Wajah '{name}' berhasil didaftarkan!")
        print(f"   Total database: {db.count()} encoding\n")


def recognize_faces():
    """Jalankan pengenalan wajah real-time."""
    db  = load_db()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera tidak ditemukan!")
        return

    print(f"\n🎭 Mode Pengenalan Wajah")
    print(f"   Database: {len(db['names'])} encoding")
    print("   Tekan 'q' untuk keluar.\n")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠ Frame kamera gagal dibaca.")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        # PCD: grayscale dan normalisasi ditampilkan sebagai analisis frame.
        gray = convert_to_grayscale_bgr(frame)
        normalized = normalize_pixels(gray)
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)

        for enc, (top, right, bottom, left) in zip(encodings, locations):
            name  = "Tidak Dikenal"
            color = (0, 80, 220)
            status = "Tidak Dikenali"
            distance_text = "-"
            cosine_text = "-"
            confidence = 0.0

            if db["encodings"]:
                # Euclidean Distance:
                # d = sqrt(sum((x_i - y_i)^2)), keputusan Dikenali jika d <= T.
                metrics = recognize_face(enc, db["encodings"], TOLERANCE)
                status = metrics.status
                distance_text = "-" if metrics.distance is None else f"{metrics.distance:.4f}"
                cosine_text = "-" if metrics.cosine_similarity is None else f"{metrics.cosine_similarity:.4f}"
                confidence = metrics.confidence
                if metrics.best_index is not None and metrics.status == "Dikenali":
                    name = db["names"][metrics.best_index]
                    color = (0, 220, 100)

            # Scale balik
            s = int(1 / SCALE)
            x1, y1, x2, y2 = left * s, top * s, right * s, bottom * s
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2), color, 2)
            info_lines = [
                f"Nama: {name}",
                f"Status: {status}",
                f"Distance: {distance_text}",
                f"Threshold: {TOLERANCE:.2f}",
                f"Cosine: {cosine_text}",
                f"Akurasi: {confidence:.1f}%",
            ]
            for idx, text in enumerate(info_lines):
                cv2.putText(frame, text, (x1 + 4, max(18, y1 - 92) + idx * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        process_time, fps = calculate_fps(start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"Process Time: {process_time:.4f}s", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"Gray mean: {np.mean(gray):.2f} | Norm mean: {np.mean(normalized):.4f}",
                    (10, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1)
        cv2.putText(frame, "Tekan 'q' untuk keluar",
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180, 180, 180), 1)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Selesai.")


def list_faces():
    db = load_db()
    if not db["names"]:
        print("Database kosong.")
        return
    from collections import Counter
    counts = Counter(db["names"])
    print(f"\n📋 Wajah Terdaftar ({len(db['names'])} total encoding):")
    for i, (name, cnt) in enumerate(sorted(counts.items()), 1):
        print(f"   {i:02d}. {name}  ({cnt} sample)")
    print()


def print_evaluation(tp: int, tn: int, fp: int, fn: int):
    metrics = evaluate_classification(tp, tn, fp, fn)
    print("\n📊 Evaluasi Sistem")
    print(f"   Accuracy : {metrics['accuracy']}%")
    print(f"   Precision: {metrics['precision']}")
    print(f"   Recall   : {metrics['recall']}")
    print(f"   F1-Score : {metrics['f1_score']}\n")


def import_dataset(dataset_dir: str):
    """Import dataset folder ke database MySQL dengan validasi gambar wajah."""
    try:
        encodings, names, warnings = load_known_faces_from_dataset(dataset_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"❌ {exc}")
        return

    db = FaceDatabase()
    for name, encoding in zip(names, encodings):
        db.add_face(name, encoding)

    for warning in warnings:
        print(f"⚠ {warning}")
    print(f"✅ Import selesai: {len(encodings)} encoding tersimpan dari dataset '{dataset_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition CLI")
    parser.add_argument("--mode", choices=["recognize", "register", "list", "import-dataset"],
                        default="recognize",
                        help="Mode: recognize | register | list | import-dataset")
    parser.add_argument("--name", type=str, default="",
                        help="Nama saat registrasi (wajib jika mode=register)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Jumlah sample saat registrasi (default: 5)")
    parser.add_argument("--eval", nargs=4, type=int, metavar=("TP", "TN", "FP", "FN"),
                        help="Hitung evaluasi opsional: --eval TP TN FP FN")
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Folder dataset untuk mode import-dataset")
    args = parser.parse_args()

    if args.eval:
        print_evaluation(*args.eval)
    elif args.mode == "register":
        if not args.name:
            print("❌ Gunakan --name 'Nama Anda' saat mode register")
        else:
            register_face(args.name, args.samples)
    elif args.mode == "list":
        list_faces()
    elif args.mode == "import-dataset":
        import_dataset(args.dataset)
    else:
        recognize_faces()
