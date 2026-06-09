# Prompt Untuk Membuat Makalah Lengkap

Gunakan prompt berikut untuk meminta AI menulis makalah lengkap berdasarkan program ini. Prompt ini sudah mencakup struktur makalah, judul, format tabel, format gambar, serta posisi peletakan gambar dan tabel.

```text
Saya memiliki program Python bernama "Enterprise Face Recognition System". Tolong buatkan makalah lengkap, formal, sistematis, dan siap dirapikan ke Microsoft Word berdasarkan deskripsi program berikut.

Judul makalah:
"Implementasi Sistem Pengenalan Wajah Real-Time Berbasis Python, OpenCV, dan Face Recognition dengan Antarmuka GUI Tkinter"

Konteks program:
- Program utama: main.py
- Program pendukung terminal/CLI: cli_mode.py
- Bahasa pemrograman: Python
- Library utama: OpenCV, face_recognition, dlib, NumPy, Pillow, Tkinter, multiprocessing, threading, JSON, logging
- Tujuan program: mengenali wajah secara real-time melalui kamera/webcam, mendaftarkan wajah baru, menghapus data wajah, menampilkan daftar wajah, dan mencatat log pengenalan wajah.
- Program GUI menggunakan Tkinter sebagai antarmuka pengguna.
- Program CLI digunakan untuk pengujian dasar melalui terminal.
- Database wajah pada program GUI disimpan dalam file face_database.json.
- Database wajah pada program CLI disimpan dalam file face_database.pkl.
- Log pengenalan wajah disimpan dalam recognition_log.txt.
- Sistem menggunakan kamera/webcam melalui cv2.VideoCapture.
- Deteksi wajah menggunakan face_recognition.face_locations dengan model HOG.
- Ekstraksi ciri wajah menggunakan face_recognition.face_encodings.
- Pencocokan wajah menggunakan face_recognition.face_distance berbasis jarak Euclidean.
- Nilai toleransi pencocokan wajah adalah 0.5.
- Frame kamera diperkecil menggunakan FRAME_SCALE 0.35 pada GUI agar proses lebih ringan.
- Program GUI memakai multiprocessing agar proses AI/pengenalan wajah berjalan pada proses terpisah dari tampilan utama.
- Program memakai threading untuk membaca kamera dan menerima hasil proses AI tanpa membekukan antarmuka.
- Program memiliki fitur SmoothBox untuk membuat kotak deteksi wajah bergerak halus dan mengurangi efek patah atau kedip pada tampilan.
- Registrasi wajah dilakukan dengan mengambil beberapa sampel wajah, lalu menghitung rata-rata encoding menggunakan NumPy.

Fitur utama program:
1. Pengenalan wajah real-time.
2. Registrasi wajah baru.
3. Penghapusan data wajah.
4. Menampilkan daftar identitas terdaftar.
5. Menampilkan log pengenalan wajah.
6. Mode CLI untuk registrasi, pengenalan, dan daftar wajah.

Ketentuan umum penulisan:
- Gunakan bahasa Indonesia baku, formal, dan mudah dipahami.
- Gunakan gaya akademik mahasiswa.
- Buat makalah cukup lengkap untuk tugas kuliah/sekolah, sekitar 12 sampai 20 halaman jika dipindahkan ke Word.
- Jangan membuat klaim berlebihan.
- Jangan mengarang angka akurasi, jumlah responden, atau hasil eksperimen yang tidak diberikan.
- Jika tidak ada data pengujian nyata, gunakan tabel skenario pengujian dengan kolom "Hasil Aktual" yang dapat diisi setelah pengujian langsung.
- Jangan terlalu banyak menampilkan kode program. Jika perlu menampilkan potongan kode, tampilkan hanya bagian yang relevan.
- Gunakan sitasi dan daftar pustaka secara konsisten.

Format umum naskah:
- Kertas: A4
- Margin: kiri 4 cm, kanan 3 cm, atas 3 cm, bawah 3 cm
- Font isi: Times New Roman 12
- Spasi: 1,5
- Perataan isi: justify
- Judul bab: huruf kapital, bold, rata tengah
- Subbab: bold, rata kiri
- Nomor halaman: bagian awal menggunakan angka romawi kecil, isi bab menggunakan angka Arab

Format peletakan gambar:
- Setiap gambar harus disebutkan terlebih dahulu di paragraf sebelum gambar muncul.
- Letakkan gambar tepat setelah paragraf yang pertama kali membahas gambar tersebut.
- Posisi gambar rata tengah.
- Nomor gambar mengikuti nomor bab, contoh: Gambar 2.1, Gambar 3.1, Gambar 4.1.
- Caption/judul gambar diletakkan di bawah gambar.
- Sumber gambar diletakkan di bawah caption.
- Jika gambar belum tersedia, gunakan placeholder "[Letakkan gambar di sini]".
- Jangan menaruh gambar tanpa penjelasan.
- Jangan menaruh gambar terlalu jauh dari paragraf pembahasannya.

Contoh format gambar:
Sebagaimana ditunjukkan pada Gambar 3.1, sistem terdiri dari kamera, proses antarmuka, proses AI worker, database wajah, dan file log.

[Letakkan Gambar 3.1 di sini]

Gambar 3.1 Arsitektur Sistem Pengenalan Wajah
Sumber: Dokumentasi penulis, 2026

Format peletakan tabel:
- Setiap tabel harus disebutkan terlebih dahulu di paragraf sebelum tabel muncul.
- Letakkan tabel tepat setelah paragraf yang pertama kali membahas data pada tabel tersebut.
- Posisi tabel rata tengah.
- Nomor tabel mengikuti nomor bab, contoh: Tabel 2.1, Tabel 3.1, Tabel 4.1.
- Judul tabel diletakkan di atas tabel.
- Sumber tabel diletakkan di bawah tabel.
- Isi tabel menggunakan font 10 atau 11 jika tabel cukup lebar.
- Gunakan tabel untuk data ringkas, bukan untuk paragraf panjang.
- Jika tabel belum memiliki hasil nyata, isi bagian hasil dengan keterangan "Diisi setelah pengujian".

Contoh format tabel:
Kebutuhan perangkat lunak yang digunakan dalam pengembangan sistem ditunjukkan pada Tabel 3.2.

Tabel 3.2 Kebutuhan Perangkat Lunak
| No | Perangkat Lunak | Keterangan |
|----|-----------------|------------|
| 1  | Python          | Bahasa pemrograman utama |
| 2  | OpenCV          | Pengolahan frame kamera |
| 3  | face_recognition | Deteksi dan encoding wajah |

Sumber: Dokumentasi penulis, 2026

Daftar gambar yang disarankan:
- Gambar 2.1 Konsep Pengolahan Citra Digital
- Gambar 2.2 Ilustrasi Proses Pengenalan Wajah
- Gambar 3.1 Arsitektur Sistem Pengenalan Wajah
- Gambar 3.2 Flowchart Proses Pengenalan Wajah
- Gambar 3.3 Flowchart Proses Registrasi Wajah
- Gambar 3.4 Rancangan Antarmuka Aplikasi GUI
- Gambar 4.1 Tampilan Aplikasi GUI
- Gambar 4.2 Tampilan Proses Registrasi Wajah
- Gambar 4.3 Tampilan Hasil Pengenalan Wajah
- Gambar 4.4 Tampilan Daftar Wajah atau Log Sistem

Daftar tabel yang disarankan:
- Tabel 2.1 Ringkasan Library yang Digunakan
- Tabel 3.1 Kebutuhan Perangkat Keras
- Tabel 3.2 Kebutuhan Perangkat Lunak
- Tabel 3.3 Kebutuhan Pengguna
- Tabel 3.4 Struktur File Program
- Tabel 3.5 Struktur Penyimpanan Database Wajah
- Tabel 4.1 Implementasi Fitur Program
- Tabel 4.2 Skenario Pengujian Sistem
- Tabel 4.3 Faktor yang Memengaruhi Hasil Pengenalan Wajah

Struktur makalah yang harus dibuat:

1. HALAMAN JUDUL
   Buat halaman judul dengan format:
   - Judul makalah
   - Nama penyusun: [isi nama]
   - NIM/Kelas: [isi NIM/Kelas]
   - Program studi: [isi program studi]
   - Nama kampus/sekolah: [isi institusi]
   - Tahun: [isi tahun]

2. ABSTRAK
   - Buat abstrak 150 sampai 250 kata.
   - Jelaskan latar belakang, tujuan, metode, fitur utama, dan hasil yang diharapkan.
   - Tambahkan kata kunci.
   - Kata kunci: pengenalan wajah, OpenCV, Python, face recognition, Tkinter.

3. KATA PENGANTAR
   - Buat kata pengantar formal dan singkat.
   - Sertakan ucapan terima kasih secara umum.
   - Jangan terlalu panjang.

4. DAFTAR ISI
   - Buat daftar isi sesuai struktur bab dan subbab.
   - Tambahkan daftar gambar dan daftar tabel jika diperlukan.

5. DAFTAR GAMBAR
   - Cantumkan daftar gambar yang digunakan dalam makalah.
   - Format contoh:
     Gambar 3.1 Arsitektur Sistem Pengenalan Wajah .......... 15

6. DAFTAR TABEL
   - Cantumkan daftar tabel yang digunakan dalam makalah.
   - Format contoh:
     Tabel 3.1 Kebutuhan Perangkat Keras .......... 13

7. BAB I PENDAHULUAN
   1.1 Latar Belakang
       Jelaskan perkembangan teknologi pengenalan wajah, kebutuhan identifikasi otomatis, dan alasan dibuatnya sistem berbasis Python dan webcam.

   1.2 Rumusan Masalah
       Buat beberapa rumusan masalah, misalnya:
       - Bagaimana merancang sistem pengenalan wajah secara real-time?
       - Bagaimana proses registrasi dan penyimpanan data wajah dilakukan?
       - Bagaimana sistem membandingkan wajah yang terdeteksi dengan database?
       - Bagaimana antarmuka GUI membantu pengguna mengelola data wajah?

   1.3 Batasan Masalah
       Cantumkan batasan:
       - Sistem menggunakan kamera/webcam.
       - Deteksi wajah menggunakan model HOG.
       - Pencocokan wajah menggunakan face_distance.
       - Database GUI menggunakan JSON.
       - Pengujian dilakukan pada kondisi perangkat dan pencahayaan terbatas.
       - Sistem belum membahas keamanan tingkat lanjut seperti liveness detection.

   1.4 Tujuan Penelitian/Perancangan
       Jelaskan tujuan membuat sistem pengenalan wajah real-time, registrasi, pengelolaan data, dan log pengenalan.

   1.5 Manfaat Penelitian/Perancangan
       Jelaskan manfaat bagi pembelajaran, keamanan dasar, presensi sederhana, dan pengembangan sistem computer vision.

   1.6 Sistematika Penulisan
       Jelaskan isi Bab I sampai Bab V secara ringkas.

8. BAB II LANDASAN TEORI
   2.1 Pengolahan Citra Digital
       Jelaskan konsep citra digital, pixel, frame, dan pemrosesan citra.
       Letakkan Gambar 2.1 setelah penjelasan konsep pengolahan citra.

   2.2 Pengenalan Wajah
       Jelaskan definisi pengenalan wajah, proses umum, dan penerapannya.
       Letakkan Gambar 2.2 setelah penjelasan tahapan pengenalan wajah.

   2.3 OpenCV
       Jelaskan fungsi OpenCV dalam membaca kamera, mengolah frame, menggambar kotak wajah, dan menampilkan hasil.

   2.4 Library face_recognition dan dlib
       Jelaskan peran face_recognition dan dlib dalam deteksi dan ekstraksi fitur wajah.

   2.5 Metode HOG untuk Deteksi Wajah
       Jelaskan HOG secara sederhana sebagai metode pendeteksian pola bentuk/gradien pada gambar.

   2.6 Face Encoding 128 Dimensi
       Jelaskan bahwa wajah direpresentasikan sebagai vektor numerik sehingga dapat dibandingkan dengan database.

   2.7 Jarak Euclidean untuk Pencocokan Wajah
       Jelaskan konsep pengukuran jarak antar-vektor. Semakin kecil jarak, semakin mirip wajah.

   2.8 Python sebagai Bahasa Pemrograman
       Jelaskan alasan Python digunakan, seperti sintaks mudah, banyak library, dan cocok untuk computer vision.

   2.9 Tkinter sebagai Antarmuka GUI
       Jelaskan penggunaan Tkinter untuk membuat jendela aplikasi, tombol, canvas, progress bar, dan dialog.

   2.10 Multiprocessing dan Threading
       Jelaskan konsep proses dan thread, serta alasan digunakan agar GUI tidak lambat saat AI memproses wajah.

   2.11 Penyimpanan Data JSON dan Pickle
       Jelaskan JSON pada GUI dan Pickle pada CLI.

   2.12 Ringkasan Library
       Buat Tabel 2.1 Ringkasan Library yang Digunakan.

9. BAB III ANALISIS DAN PERANCANGAN SISTEM
   3.1 Analisis Kebutuhan Sistem
       Buat penjelasan kebutuhan sistem.
       Sertakan:
       - Tabel 3.1 Kebutuhan Perangkat Keras
       - Tabel 3.2 Kebutuhan Perangkat Lunak
       - Tabel 3.3 Kebutuhan Pengguna

   3.2 Perancangan Arsitektur Sistem
       Jelaskan hubungan kamera, GUI, AI worker process, database wajah, dan file log.
       Letakkan Gambar 3.1 Arsitektur Sistem Pengenalan Wajah setelah penjelasan arsitektur.

   3.3 Alur Kerja Sistem
       Jelaskan alur:
       - Program dijalankan.
       - Kamera dibuka.
       - Frame kamera dibaca.
       - Frame diperkecil.
       - Frame dikirim ke AI worker.
       - Wajah dideteksi.
       - Encoding wajah dihitung.
       - Encoding dibandingkan dengan database.
       - Nama dan confidence ditampilkan.
       - Log disimpan jika wajah dikenali.
       Letakkan Gambar 3.2 Flowchart Proses Pengenalan Wajah setelah penjelasan alur kerja.

   3.4 Alur Registrasi Wajah
       Jelaskan proses:
       - Pengguna menekan tombol Daftarkan Wajah.
       - Pengguna memasukkan nama.
       - Sistem mengambil beberapa sampel wajah.
       - Encoding dihitung.
       - Rata-rata encoding disimpan ke database.
       - AI worker memuat ulang database.
       Letakkan Gambar 3.3 Flowchart Proses Registrasi Wajah setelah penjelasan registrasi.

   3.5 Perancangan Database
       Jelaskan struktur face_database.json yang berisi names dan encodings.
       Buat Tabel 3.5 Struktur Penyimpanan Database Wajah.

   3.6 Perancangan Struktur File
       Jelaskan file:
       - main.py
       - cli_mode.py
       - requirements.txt
       - face_database.json
       - face_database.pkl
       - recognition_log.txt
       Buat Tabel 3.4 Struktur File Program.

   3.7 Perancangan Antarmuka
       Jelaskan bagian GUI:
       - Area kamera/canvas
       - Status sistem
       - Tombol Daftarkan Wajah
       - Tombol Hapus Wajah
       - Tombol Daftar Wajah
       - Tombol Lihat Log
       - Progress bar registrasi
       Letakkan Gambar 3.4 Rancangan Antarmuka Aplikasi GUI setelah penjelasan antarmuka.

10. BAB IV IMPLEMENTASI DAN PENGUJIAN
   4.1 Implementasi Program
       Jelaskan implementasi main.py sebagai aplikasi GUI dan cli_mode.py sebagai mode terminal.
       Buat Tabel 4.1 Implementasi Fitur Program.

   4.2 Implementasi Deteksi dan Pengenalan Wajah
       Jelaskan penggunaan:
       - face_recognition.face_locations
       - face_recognition.face_encodings
       - face_recognition.face_distance
       - TOLERANCE 0.5
       Letakkan Gambar 4.3 Tampilan Hasil Pengenalan Wajah setelah penjelasan jika gambar tersedia.

   4.3 Implementasi Registrasi Wajah
       Jelaskan pengambilan sampel, jeda antar sampel, perhitungan rata-rata encoding, dan penyimpanan database.
       Letakkan Gambar 4.2 Tampilan Proses Registrasi Wajah setelah penjelasan jika gambar tersedia.

   4.4 Implementasi GUI
       Jelaskan penggunaan Tkinter, Canvas, tombol kontrol, progress bar, status sistem, dan tampilan kamera.
       Letakkan Gambar 4.1 Tampilan Aplikasi GUI setelah penjelasan GUI jika gambar tersedia.

   4.5 Implementasi Multiprocessing dan Threading
       Jelaskan bahwa proses AI dipisahkan dari proses GUI menggunakan multiprocessing. Jelaskan juga threading untuk kamera dan penerimaan hasil agar antarmuka tetap responsif.

   4.6 Implementasi SmoothBox
       Jelaskan fungsi SmoothBox untuk memperhalus perpindahan kotak wajah dan mengurangi tampilan patah/kedip.

   4.7 Implementasi Log Sistem
       Jelaskan penyimpanan log ke recognition_log.txt saat wajah dikenali.
       Letakkan Gambar 4.4 Tampilan Daftar Wajah atau Log Sistem setelah penjelasan jika gambar tersedia.

   4.8 Pengujian Sistem
       Buat Tabel 4.2 Skenario Pengujian Sistem dengan kolom:
       No, Skenario Pengujian, Langkah Pengujian, Hasil yang Diharapkan, Hasil Aktual, Status.

       Sertakan skenario:
       - Membuka aplikasi GUI.
       - Kamera berhasil terdeteksi.
       - Registrasi wajah baru.
       - Pengenalan wajah yang sudah terdaftar.
       - Pengenalan wajah yang belum terdaftar.
       - Menghapus data wajah.
       - Menampilkan daftar wajah.
       - Menampilkan log.
       - Menjalankan mode CLI register.
       - Menjalankan mode CLI recognize.
       - Menjalankan mode CLI list.

       Untuk kolom Hasil Aktual, jika belum ada hasil nyata, isi dengan "Diisi setelah pengujian langsung".

   4.9 Analisis Hasil Pengujian
       Jelaskan faktor yang memengaruhi hasil pengenalan wajah, seperti pencahayaan, posisi wajah, kualitas kamera, jarak wajah, jumlah sampel wajah, dan nilai tolerance.
       Buat Tabel 4.3 Faktor yang Memengaruhi Hasil Pengenalan Wajah.

11. BAB V PENUTUP
   5.1 Kesimpulan
       Buat kesimpulan berdasarkan tujuan dan implementasi program.
       Jelaskan bahwa sistem mampu melakukan registrasi, pengenalan, pengelolaan data, dan pencatatan log.

   5.2 Saran Pengembangan
       Sertakan saran:
       - Menambahkan login admin.
       - Menambahkan enkripsi database.
       - Menggunakan database SQL.
       - Menambahkan laporan rekap kehadiran.
       - Menambahkan model deteksi CNN/GPU jika perangkat mendukung.
       - Menambahkan liveness detection untuk mencegah penggunaan foto.
       - Menambahkan ekspor log ke Excel/PDF.

12. DAFTAR PUSTAKA
   - Buat daftar pustaka minimal 5 sumber.
   - Gunakan format APA atau IEEE secara konsisten.
   - Sertakan referensi tentang OpenCV, dlib, face_recognition, Python, Tkinter, dan pengolahan citra.
   - Prioritaskan dokumentasi resmi.
   - Jangan membuat referensi palsu.

13. LAMPIRAN
   Lampiran A. Struktur File Program
   Cantumkan:
   - main.py
   - cli_mode.py
   - requirements.txt
   - face_database.json
   - face_database.pkl
   - recognition_log.txt

   Lampiran B. Daftar Dependency
   Cantumkan dependency:
   - opencv-python==4.9.0.80
   - opencv-contrib-python==4.9.0.80
   - numpy==1.26.4
   - Pillow==10.3.0
   - face-recognition==1.3.0
   - dlib==19.24.2
   - cmake==3.29.3

   Lampiran C. Contoh Perintah Menjalankan Program
   Cantumkan:
   - python main.py
   - python cli_mode.py --mode recognize
   - python cli_mode.py --mode register --name "Nama Anda" --samples 5
   - python cli_mode.py --mode list

   Lampiran D. Potongan Kode Penting
   Jika perlu, tampilkan potongan kode singkat untuk:
   - Konfigurasi global
   - Deteksi wajah
   - Pencocokan wajah
   - Penyimpanan database
   - Penulisan log

Tambahan instruksi khusus:
- Buat tulisan mengalir seperti makalah yang sudah jadi, bukan hanya kerangka.
- Tetap berikan placeholder gambar jika gambar belum tersedia.
- Jangan menempatkan semua gambar di akhir bab; letakkan dekat dengan pembahasannya.
- Jangan menempatkan semua tabel di lampiran; tabel utama tetap berada di bab yang relevan.
- Pastikan setiap gambar dan tabel dirujuk di dalam teks.
- Buat judul gambar dan tabel jelas, singkat, dan akademik.
- Gunakan sumber "Dokumentasi penulis, 2026" untuk gambar/tabel yang berasal dari program sendiri.
```
