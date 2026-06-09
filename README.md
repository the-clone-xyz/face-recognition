# 🎭 Face Recognition System
## Python 3.12 | Windows 10

---

## 📁 Struktur File

```
face_recognition_project/
├── main.py           # Aplikasi utama dengan GUI (tkinter)
├── cli_mode.py       # Versi CLI tanpa GUI (untuk testing)
├── requirements.txt  # Daftar library
└── README.md         # Panduan ini
```

---

## ⚙️ Instalasi Codespace

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip "setuptools<81" wheel
.venv/bin/python -m pip install -r requirements.txt
docker compose up -d
```

phpMyAdmin berjalan di port `8080`.

Login default:

| Field | Nilai |
|-------|-------|
| Server | `mysql` |
| Username | `facerec_user` |
| Password | `facerec_password` |
| Database | `facerecognition` |

Konfigurasi koneksi aplikasi bisa diubah lewat environment variable:

```bash
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=facerec_user
DB_PASSWORD=facerec_password
DB_NAME=facerecognition
```

Untuk migrasi database lama ke MySQL:

```bash
.venv/bin/python migrate_local_db.py face_database.pkl
```

---

## ⚙️ Instalasi Windows Lama (Ikuti Urutan Ini!)

### Langkah 1 — Pastikan Build Tools tersedia

Library `dlib` (dependensi `face-recognition`) butuh compiler C++.

1. Download **Visual Studio Build Tools 2022**:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Saat install, centang **"Desktop development with C++"**
3. Restart PC setelah selesai

### Langkah 2 — Install CMake

```
pip install cmake
```

### Langkah 3 — Install dlib (via wheel, lebih mudah di Windows)

Daripada compile dari source, gunakan pre-built wheel:

```
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp312-cp312-win_amd64.whl
```

> ✅ Ini adalah cara **termudah** untuk install dlib di Python 3.12 Windows 10.
> Jika link berubah, cari "dlib wheel Python 3.12 Windows" di GitHub.

### Langkah 4 — Install library lainnya

```
pip install face-recognition opencv-python opencv-contrib-python numpy Pillow
```

### Langkah 5 — Verifikasi instalasi

```python
python -c "import face_recognition; import cv2; print('✅ Semua library OK!')"
```

---

## 🚀 Cara Menjalankan

### GUI (Direkomendasikan)

```
python main.py
```

### CLI (Untuk testing)

```bash
# Mode pengenalan (default)
python cli_mode.py

# Daftarkan wajah baru
python cli_mode.py --mode register --name "Budi" --samples 5

# Lihat daftar wajah
python cli_mode.py --mode list
```

---

## 🖥️ Fitur GUI

| Tombol | Fungsi |
|--------|--------|
| ➕ Daftarkan Wajah | Registrasi wajah baru (otomatis ambil 5 sample) |
| 🗑 Hapus Wajah | Hapus wajah dari database |
| 📋 Daftar Wajah | Lihat semua wajah terdaftar |
| 📄 Lihat Log | Riwayat pengenalan wajah |

---

## ⚙️ Konfigurasi (di main.py)

```python
DB_FILE       = "face_database.pkl"  # Lokasi file database
TOLERANCE     = 0.5                  # 0.4 = ketat, 0.6 = longgar
FRAME_SCALE   = 0.5                  # 0.5 = setengah resolusi (lebih cepat)
reg_target    = 5                    # Jumlah sample saat registrasi
```

---

## 🔍 Cara Kerja

```
Webcam Frame
    │
    ▼
Resize (50%) ──► Hemat CPU
    │
    ▼
Deteksi Lokasi Wajah (HOG)
    │
    ▼
Ekstrak 128-dimensi Face Encoding
    │
    ▼
Bandingkan dengan Database (Euclidean Distance)
    │
    ├── Jarak ≤ TOLERANCE → Dikenal ✅
    └── Jarak > TOLERANCE → Tidak Dikenal ❌
```

---

## 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'face_recognition'`
→ Pastikan dlib sudah terinstall duluan sebelum face-recognition.

### `dlib build failed`
→ Gunakan pre-built wheel (Langkah 3 di atas).

### Kamera tidak terdeteksi
→ Coba ganti `cv2.VideoCapture(0)` menjadi `cv2.VideoCapture(1)` di main.py.

### Wajah sering tidak dikenal
→ Kurangi nilai `TOLERANCE` (misal dari 0.5 ke 0.45).

### Performa lambat / FPS rendah
→ Kurangi nilai `FRAME_SCALE` (misal dari 0.5 ke 0.35).

---

## 📦 File yang Dibuat Otomatis

| File | Keterangan |
|------|-----------|
| `face_database.pkl` | Database encoding wajah (dibuat saat registrasi pertama) |
| `recognition_log.txt` | Log pengenalan (dibuat otomatis) |
