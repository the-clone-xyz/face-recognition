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
import pickle
import os
import argparse
import time

DB_FILE   = "face_database.pkl"
TOLERANCE = 0.5
SCALE     = 0.5


def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}


def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def register_face(name: str, n_samples: int = 5):
    """Ambil n_samples encoding wajah dan simpan ke database."""
    db  = load_db()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kamera tidak ditemukan!")
        return

    print(f"\n📸 Mendaftarkan wajah: {name}")
    print(f"   Posisikan wajah Anda di depan kamera. Butuh {n_samples} sample.")
    print("   Tekan 'q' untuk batal.\n")

    encodings  = []
    last_time  = 0

    while len(encodings) < n_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        frame     = cv2.flip(frame, 1)
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        for (top, right, bottom, left) in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 255), 2)

        info = f"Sample: {len(encodings)}/{n_samples}"
        cv2.putText(frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
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
        db["encodings"].append(avg)
        db["names"].append(name)
        save_db(db)
        print(f"\n✅ Wajah '{name}' berhasil didaftarkan!")
        print(f"   Total database: {len(db['names'])} encoding\n")


def recognize_faces():
    """Jalankan pengenalan wajah real-time."""
    db  = load_db()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Kamera tidak ditemukan!")
        return

    print(f"\n🎭 Mode Pengenalan Wajah")
    print(f"   Database: {len(db['names'])} encoding")
    print("   Tekan 'q' untuk keluar.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)

        for enc, (top, right, bottom, left) in zip(encodings, locations):
            name  = "Tidak Dikenal"
            color = (0, 80, 220)

            if db["encodings"]:
                dists    = face_recognition.face_distance(db["encodings"], enc)
                best_idx = int(np.argmin(dists))
                if dists[best_idx] <= TOLERANCE:
                    name  = db["names"][best_idx]
                    conf  = round((1 - dists[best_idx]) * 100, 1)
                    name  = f"{name} ({conf}%)"
                    color = (0, 220, 100)

            # Scale balik
            s = int(1 / SCALE)
            cv2.rectangle(frame,
                          (left * s, top * s),
                          (right * s, bottom * s), color, 2)
            cv2.putText(frame, name,
                        (left * s + 4, bottom * s + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "Tekan 'q' untuk keluar",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition CLI")
    parser.add_argument("--mode", choices=["recognize", "register", "list"],
                        default="recognize",
                        help="Mode: recognize | register | list")
    parser.add_argument("--name", type=str, default="",
                        help="Nama saat registrasi (wajib jika mode=register)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Jumlah sample saat registrasi (default: 5)")
    args = parser.parse_args()

    if args.mode == "register":
        if not args.name:
            print("❌ Gunakan --name 'Nama Anda' saat mode register")
        else:
            register_face(args.name, args.samples)
    elif args.mode == "list":
        list_faces()
    else:
        recognize_faces()
