-- Database utama untuk menyimpan data wajah yang sudah diregistrasi.
CREATE DATABASE IF NOT EXISTS facerecognition
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- User khusus aplikasi; root XAMPP tidak dipakai oleh aplikasi Flask.
CREATE USER IF NOT EXISTS 'facerec_user'@'localhost'
    IDENTIFIED BY 'facerec_password';
CREATE USER IF NOT EXISTS 'facerec_user'@'127.0.0.1'
    IDENTIFIED BY 'facerec_password';

-- Jika user sudah pernah dibuat, password disamakan lagi dengan konfigurasi face_db.py.
ALTER USER 'facerec_user'@'localhost'
    IDENTIFIED BY 'facerec_password';
ALTER USER 'facerec_user'@'127.0.0.1'
    IDENTIFIED BY 'facerec_password';

-- Hak akses dibatasi hanya ke database facerecognition.
GRANT ALL PRIVILEGES ON facerecognition.* TO 'facerec_user'@'localhost';
GRANT ALL PRIVILEGES ON facerecognition.* TO 'facerec_user'@'127.0.0.1';
FLUSH PRIVILEGES;

USE facerecognition;

-- Tabel faces menyimpan satu nama dan satu vektor encoding wajah dalam format JSON.
CREATE TABLE IF NOT EXISTS faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    encoding JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_faces_name (name)
);
