# Face Recognition Web App

Project ini dijalankan sebagai aplikasi web lokal dengan Python 3.12 dan MySQL dari XAMPP.

## Struktur Utama

```text
web_app.py          Aplikasi Flask
face_db.py          Koneksi dan schema MySQL
pcd_utils.py        Utilitas pengolahan citra dan pencocokan wajah
templates/          Halaman web
static/             CSS dan JavaScript
requirements.txt    Dependency runtime Python
xampp_setup.sql     Setup database XAMPP
```

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start MySQL dari XAMPP Control Panel, lalu import `xampp_setup.sql` lewat phpMyAdmin:

```text
http://localhost/phpmyadmin
```

Database default:

```text
Host      : 127.0.0.1
Port      : 3306
Database  : facerecognition
User      : facerec_user
Password  : facerec_password
```

## Jalankan

```powershell
.\.venv\Scripts\python.exe web_app.py
```

Buka aplikasi:

```text
http://127.0.0.1:5001
```

Halaman database internal:

```text
http://127.0.0.1:5001/database
```

## Verifikasi

```powershell
.\.venv\Scripts\python.exe -c "import flask, mysql.connector, cv2, face_recognition, dlib, PIL, numpy; print('OK')"
```

## Konfigurasi Opsional

Koneksi database bisa diubah lewat environment variable:

```powershell
$env:DB_HOST="127.0.0.1"
$env:DB_PORT="3306"
$env:DB_USER="facerec_user"
$env:DB_PASSWORD="facerec_password"
$env:DB_NAME="facerecognition"
```
