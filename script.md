# Script Presentasi — Face Recognition Web App

> **Mata Kuliah:** Pengolahan Citra Digital  
> **Anggota Tim:** Yogi · Dian · Tyo · Nazib

---

## Pembagian Presentasi

| Bagian | Presenter | Topik |
|--------|-----------|-------|
| 1 | **Yogi** | Pendahuluan & Arsitektur Sistem |
| 2 | **Dian** | Pengolahan Citra Digital (PCD) |
| 3 | **Tyo** | Face Recognition & Algoritma Pencocokan |
| 4 | **Nazib** | Database, Demo Aplikasi & Penutup |

---

## Bagian 1 — Yogi: Pendahuluan & Arsitektur Sistem

### Slide 1: Judul & Perkenalan

> "Assalamualaikum warahmatullahi wabarakatuh. Perkenalkan, kami dari kelompok [nama kelompok]. Hari ini kami akan mempresentasikan project **Face Recognition Web App** berbasis Pengolahan Citra Digital. Saya Yogi, dan bersama saya ada Dian, Tyo, dan Nazib."

### Slide 2: Latar Belakang

> "Sistem face recognition sudah banyak digunakan di dunia nyata, mulai dari absensi otomatis, keamanan gedung, hingga verifikasi identitas digital. Project kami mengimplementasikan konsep-konsep **Pengolahan Citra Digital** secara langsung ke dalam sebuah aplikasi web yang dapat melakukan registrasi wajah dan mengenali wajah secara **realtime** melalui kamera."

### Slide 3: Tujuan Project

> "Tujuan dari project ini adalah:
> 1. Mengimplementasikan teknik pengolahan citra seperti **Grayscale Conversion**, **Normalisasi Piksel**, dan **Resizing** dalam konteks face recognition.
> 2. Menerapkan algoritma **Euclidean Distance** dan **Cosine Similarity** untuk pencocokan wajah.
> 3. Membuat aplikasi web interaktif yang menampilkan proses pengolahan citra secara visual dan realtime."

### Slide 4: Arsitektur Sistem

> "Secara keseluruhan, project kami memiliki arsitektur seperti ini:"

```
┌─────────────────────────────────────────────────┐
│              Browser (Frontend)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Kamera  │  │  Canvas  │  │  Histogram   │  │
│  │  WebRTC  │  │  Overlay │  │  RGB Chart   │  │
│  └────┬─────┘  └──────────┘  └──────────────┘  │
│       │ frame base64                            │
└───────┼─────────────────────────────────────────┘
        │  HTTP POST /api/recognize
        │  HTTP POST /api/register
        ▼
┌─────────────────────────────────────────────────┐
│           Flask Server (web_app.py)             │
│  ┌──────────────────┐  ┌─────────────────────┐  │
│  │   pcd_utils.py   │  │    face_db.py       │  │
│  │  Grayscale       │  │  MySQL Connector    │  │
│  │  Normalisasi     │  │  CRUD Encoding      │  │
│  │  Euclidean Dist. │  │  JSON ↔ Numpy       │  │
│  │  Cosine Sim.     │  │                     │  │
│  └──────────────────┘  └────────┬────────────┘  │
└─────────────────────────────────┼───────────────┘
                                  │
                                  ▼
                    ┌──────────────────────┐
                    │  MySQL / MariaDB     │
                    │  (XAMPP)             │
                    │  Database:           │
                    │  facerecognition     │
                    │  Tabel: faces        │
                    └──────────────────────┘
```

> "Frontend menggunakan **HTML, CSS, dan JavaScript** yang berjalan di browser. Kamera diakses melalui **WebRTC**, lalu setiap frame dikirim ke backend **Flask** dalam format base64. Backend melakukan pengolahan citra menggunakan **OpenCV** dan **face_recognition library**, lalu hasilnya dikirim kembali ke browser."

### Slide 5: Teknologi yang Digunakan

> "Berikut teknologi yang kami gunakan:"

| Komponen | Teknologi |
|----------|-----------|
| Backend | Python 3.12, Flask 3.0.3 |
| Computer Vision | OpenCV 4.9, face_recognition 1.3, dlib |
| Database | MySQL/MariaDB (XAMPP) |
| Frontend | HTML5, CSS3, JavaScript (Vanilla) |
| Kamera | WebRTC API |
| Komunikasi | REST API (JSON) |

> "Selanjutnya, Dian akan menjelaskan bagian inti dari project ini yaitu **Pengolahan Citra Digital**."

---

## Bagian 2 — Dian: Pengolahan Citra Digital (PCD)

### Slide 6: Pipeline Pengolahan Citra

> "Terima kasih Yogi. Saya Dian, akan menjelaskan bagaimana pengolahan citra digital diterapkan dalam project kami. Berikut adalah tahapan yang terjadi setiap kali kamera mengirimkan frame:"

```
Frame RGB → Grayscale → Normalisasi → Resize → Deteksi Wajah → Face Encoding
```

> "Setiap tahapan ini divisualisasikan secara realtime di aplikasi kami melalui panel **Tahapan Citra Realtime**."

### Slide 7: Konversi Grayscale

> "Tahap pertama adalah konversi dari citra berwarna **RGB** menjadi **Grayscale**. Kami menggunakan rumus luminance standar:"

```
I(x,y) = 0.299R + 0.587G + 0.114B
```

> "Bobot ini bukan sembarang, tapi didasarkan pada sensitivitas mata manusia dimana mata lebih peka terhadap warna hijau (0.587) dibanding merah (0.299) dan biru (0.114). Dalam kode, ini diimplementasikan di file `pcd_utils.py`:"

```python
def convert_to_grayscale_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
```

> "OpenCV secara internal menerapkan rumus luminance yang sama saat melakukan konversi."

### Slide 8: Normalisasi Piksel

> "Setelah grayscale, nilai piksel yang berada di rentang 0-255 dinormalisasi ke rentang 0 sampai 1 menggunakan rumus:"

```
I'(x,y) = I(x,y) / 255
```

> "Normalisasi ini penting agar perhitungan numerik lebih stabil dan konsisten. Dalam kode:"

```python
def normalize_pixels(gray_image: np.ndarray) -> np.ndarray:
    return gray_image.astype(np.float32) / 255.0
```

### Slide 9: Resize Frame

> "Sebelum deteksi wajah, frame diperkecil dengan skala **0.35** (35%) dari ukuran asli. Ini dilakukan untuk mempercepat proses deteksi tanpa mengorbankan akurasi secara signifikan."

```python
FRAME_SCALE = 0.35
small = cv2.resize(rgb_image, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
```

> "Contohnya, jika resolusi kamera 640×480 piksel, maka frame yang diproses hanya 224×168 piksel. Ini mengurangi jumlah piksel yang harus diproses menjadi hanya sekitar **12%** dari aslinya, sehingga proses deteksi jauh lebih cepat."

### Slide 10: Histogram RGB

> "Di aplikasi kami, histogram citra RGB ditampilkan secara realtime. Histogram menunjukkan **distribusi intensitas** setiap kanal warna (Red, Green, Blue) dalam rentang 0-255. Dari histogram, kita bisa melihat apakah citra terlalu terang, terlalu gelap, atau memiliki distribusi yang baik. Ini membantu kita memahami karakteristik citra yang sedang diproses."

> "Selanjutnya, Tyo akan menjelaskan bagaimana wajah dideteksi dan dikenali menggunakan algoritma pencocokan."

---

## Bagian 3 — Tyo: Face Recognition & Algoritma Pencocokan

### Slide 11: Deteksi Wajah dengan HOG

> "Terima kasih Dian. Saya Tyo, akan menjelaskan proses face recognition. Tahap pertama setelah preprocessing adalah **deteksi wajah**. Kami menggunakan model **HOG (Histogram of Oriented Gradients)** dari library `face_recognition`."

```python
locations = face_recognition.face_locations(small, model="hog")
encodings = face_recognition.face_encodings(small, locations)
```

> "Model HOG bekerja dengan mendeteksi pola gradien pada citra yang membentuk struktur wajah. Hasilnya berupa koordinat **bounding box** (top, right, bottom, left) dari setiap wajah yang terdeteksi. Selanjutnya, untuk setiap wajah yang ditemukan, library menghasilkan **face encoding** berupa **vektor 128 dimensi** yang merepresentasikan fitur unik wajah tersebut."

### Slide 12: Euclidean Distance

> "Setelah mendapatkan encoding dari wajah yang terdeteksi, langkah berikutnya adalah **mencocokkan** encoding tersebut dengan data wajah yang sudah tersimpan di database. Kami menggunakan **Euclidean Distance** sebagai metrik utama:"

```
d = √(Σᵢ₌₁ⁿ (xᵢ - yᵢ)²)
```

> "Dimana **x** adalah encoding wajah dari kamera dan **y** adalah encoding dari database. Semakin kecil nilai d, maka semakin mirip kedua wajah tersebut. Dalam kode:"

```python
def calculate_euclidean_distance(current_encoding, known_encoding):
    return float(np.linalg.norm(current_encoding - known_encoding))
```

### Slide 13: Cosine Similarity

> "Sebagai metrik tambahan, kami juga menghitung **Cosine Similarity**:"

```
cos(θ) = (A · B) / (|A| × |B|)
```

> "Cosine Similarity mengukur kemiripan berdasarkan **arah vektor**, bukan jarak. Nilainya berkisar 0 sampai 1, dimana 1 berarti identik. Metrik ini bersifat komplementer; Euclidean Distance menjadi keputusan utama, sedangkan Cosine Similarity menjadi informasi pendukung."

```python
def calculate_cosine_similarity(vector_a, vector_b):
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
```

### Slide 14: Logika Keputusan

> "Keputusan akhir apakah wajah **Dikenali** atau **Tidak Dikenali** ditentukan menggunakan **threshold** (nilai ambang batas):"

```
Status = Dikenali,       jika d ≤ T (Threshold = 0.55)
Status = Tidak Dikenali, jika d > T
```

> "Threshold kami set ke **0.55**. Artinya, jika jarak Euclidean antara encoding kamera dan encoding terdekat di database kurang dari atau sama dengan 0.55, maka wajah tersebut **dikenali**. Dalam kode:"

```python
TOLERANCE = 0.55
status = "Dikenali" if best_distance <= threshold else "Tidak Dikenali"
confidence = max(0.0, min(100.0, (1.0 - best_distance) * 100.0))
```

> "Kami juga menghitung **confidence** (tingkat keyakinan) berdasarkan jarak. Semakin kecil jarak, semakin tinggi confidence-nya."

### Slide 15: Proses Registrasi Wajah

> "Untuk meregistrasi wajah baru, aplikasi mengambil **beberapa frame** (5 sample), mengekstrak encoding dari setiap frame, lalu menghitung **rata-rata encoding** sebagai representasi final wajah."

```python
avg_encoding = np.mean(collected, axis=0)
db.add_face(name, avg_encoding)
```

> "Rata-rata ini membuat data wajah lebih stabil dibanding hanya menggunakan satu frame saja, karena setiap frame bisa memiliki sedikit variasi ekspresi dan posisi. Selanjutnya Nazib akan menjelaskan bagian database dan melakukan demo aplikasi."

---

## Bagian 4 — Nazib: Database, Demo Aplikasi & Penutup

### Slide 16: Struktur Database

> "Terima kasih Tyo. Saya Nazib, akan menjelaskan bagian penyimpanan data dan melakukan demo. Kami menggunakan **MySQL/MariaDB** dari XAMPP dengan database bernama `facerecognition`. Tabel utamanya adalah `faces`:"

```sql
CREATE TABLE IF NOT EXISTS faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    encoding JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_faces_name (name)
);
```

> "Setiap baris menyimpan:
> - **id**: primary key auto-increment
> - **name**: nama orang yang diregistrasi
> - **encoding**: vektor 128 dimensi disimpan dalam format **JSON**
> - **created_at**: waktu registrasi"

### Slide 17: Alur Data (CRUD)

> "File `face_db.py` menangani semua operasi database. Saat aplikasi dimulai, tabel otomatis dibuat jika belum ada. Berikut alur datanya:"

```
Registrasi:
  Frame kamera → Encoding → JSON → INSERT ke MySQL

Pengenalan:
  Semua encoding di-load dari MySQL → Numpy array → Bandingkan dengan kamera

Hapus Wajah:
  DELETE FROM faces WHERE name = 'nama'
```

> "Encoding disimpan sebagai JSON di MySQL, dan saat di-load kembali, dikonversi menjadi numpy array untuk perhitungan jarak."

```python
class FaceDatabase:
    def add_face(self, name, encoding):
        encoding_json = json.dumps(encoding.tolist())
        # INSERT INTO faces (name, encoding) VALUES (%s, %s)

    def load(self):
        # SELECT name, encoding FROM faces
        # Konversi JSON → numpy array
```

### Slide 18: REST API Endpoints

> "Backend menyediakan beberapa **REST API** untuk komunikasi antara frontend dan backend:"

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| GET | `/` | Halaman utama (kamera + recognition) |
| GET | `/database` | Halaman tabel data wajah |
| GET | `/api/status` | Cek status koneksi database |
| GET | `/api/faces` | Daftar wajah terdaftar |
| POST | `/api/recognize` | Kirim frame → terima hasil deteksi |
| POST | `/api/register` | Registrasi wajah baru |
| DELETE | `/api/faces/<name>` | Hapus data wajah |

### Slide 19: Demo Aplikasi

> "Sekarang saya akan melakukan demo langsung. Pertama, kita jalankan MySQL dari XAMPP, lalu jalankan aplikasi Flask."

**Langkah Demo:**

1. **Jalankan Aplikasi**
   ```powershell
   cd "C:\Users\User\Desktop\face-recognition"
   .\.venv\Scripts\python.exe web_app.py
   ```
   Buka browser: `http://127.0.0.1:5001`

2. **Nyalakan Kamera** → Klik tombol "Nyalakan Kamera"

3. **Tunjukkan Pipeline PCD** → Perhatikan panel "Tahapan Citra Realtime" yang menampilkan:
   - Frame RGB asli
   - Hasil Grayscale
   - Hasil Resize
   - Hasil Deteksi

4. **Registrasi Wajah** → Masukkan nama, klik "Daftarkan", lalu tunggu 5 sample terambil

5. **Mulai Deteksi** → Klik "Mulai Deteksi" untuk melihat pengenalan wajah realtime. Perhatikan:
   - Kotak wajah muncul di overlay
   - Status: Dikenali / Tidak Dikenali
   - Nilai Distance, Cosine Similarity, dan Confidence di panel Angka & Rumus

6. **Halaman Database** → Buka `/database` untuk melihat data wajah yang tersimpan di MySQL

### Slide 20: Metrik yang Ditampilkan

> "Selama demo berjalan, aplikasi menampilkan berbagai metrik realtime:"

| Metrik | Keterangan |
|--------|------------|
| Resolusi | Ukuran frame kamera asli |
| Piksel/frame | Jumlah total piksel per frame |
| Resize | Persentase pengecilan frame |
| Beban proses | Perbandingan ukuran resize vs asli |
| FPS Pipeline | Frame per second di sisi frontend |
| FPS AI | Frame per second pemrosesan backend |
| Waktu proses | Lama proses per frame (detik) |
| Distance | Jarak Euclidean ke encoding terdekat |
| Cosine Similarity | Kemiripan arah vektor |
| Confidence | Tingkat keyakinan (%) |

### Slide 21: Kesimpulan

> "Sebagai kesimpulan dari project kami:
> 1. Kami berhasil mengimplementasikan konsep **Pengolahan Citra Digital** (grayscale, normalisasi, resize) dalam sistem face recognition yang berjalan secara realtime.
> 2. Algoritma **Euclidean Distance** terbukti efektif sebagai metrik utama pencocokan wajah dengan didukung **Cosine Similarity** sebagai metrik tambahan.
> 3. Sistem kami dapat melakukan **registrasi** dan **pengenalan** wajah dengan antarmuka web yang menampilkan seluruh proses secara visual dan transparan.
> 4. Penggunaan **MySQL** sebagai penyimpanan encoding memastikan data wajah bersifat **persisten** dan dapat dikelola melalui phpMyAdmin."

### Slide 22: Penutup

> "Demikian presentasi kami tentang Face Recognition Web App berbasis Pengolahan Citra Digital. Apakah ada pertanyaan?"

> "Terima kasih atas perhatiannya. Wassalamualaikum warahmatullahi wabarakatuh."

---

## Ringkasan Pembagian

| Presenter | Slide | Durasi (±) | Topik Utama |
|-----------|-------|------------|-------------|
| **Yogi** | 1–5 | 5 menit | Pendahuluan, arsitektur, teknologi |
| **Dian** | 6–10 | 5 menit | Grayscale, normalisasi, resize, histogram |
| **Tyo** | 11–15 | 7 menit | HOG, Euclidean Distance, Cosine Similarity, keputusan, registrasi |
| **Nazib** | 16–22 | 8 menit | Database, API, demo langsung, kesimpulan |

> **Total estimasi: ±25 menit** (termasuk demo)
