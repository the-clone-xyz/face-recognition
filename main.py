"""
Enterprise Face Recognition System (Final Production Release)
===============================================================
Author: Senior Software & Security Engineer
Description: Sistem pengenalan wajah multi-proses yang aman, 
sangat mulus (60 FPS UI), dan akurat.
"""

import multiprocessing as mp
import queue
import cv2
import face_recognition
import numpy as np
import json
import os
import threading
import time
import math
import logging
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime

# ──────────────────────────────────────────────
# KONFIGURASI LOGGING TERMINAL
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("FaceRecSys")

# ──────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ──────────────────────────────────────────────
DB_FILE        = "face_database.json"
LOG_FILE       = "recognition_log.txt"
TOLERANCE      = 0.5    
FRAME_SCALE    = 0.35   
UNKNOWN_LABEL  = "Tidak Dikenal"
UI_REFRESH_MS  = 15     # Target ~60 FPS Render
SMOOTH_SPEED   = 0.35   # Kecepatan animasi Lerp kotak pelacak
GRACE_PERIOD   = 1.5    # Toleransi wajah hilang sebelum kotak dihapus
TRACK_RADIUS   = 250    # Jarak toleransi pergerakan wajah antar-frame

# ──────────────────────────────────────────────
# KELAS ANIMASI & TRACKING (Anti-Patah & Anti-Kedip)
# ──────────────────────────────────────────────
class SmoothBox:
    def __init__(self, top, right, bottom, left, name, conf):
        self.top, self.right, self.bottom, self.left = top, right, bottom, left
        self.t_top, self.t_right, self.t_bottom, self.t_left = top, right, bottom, left
        self.name = name
        self.conf = conf
        self.last_seen = time.time()

    def update_target(self, top, right, bottom, left, name, conf):
        self.t_top, self.t_right, self.t_bottom, self.t_left = top, right, bottom, left
        if name != UNKNOWN_LABEL or self.name == UNKNOWN_LABEL:
            self.name = name
            self.conf = conf
        self.last_seen = time.time()

    def glide(self):
        # Linear Interpolation (Lerp) untuk pergerakan halus
        self.top += (self.t_top - self.top) * SMOOTH_SPEED
        self.right += (self.t_right - self.right) * SMOOTH_SPEED
        self.bottom += (self.t_bottom - self.bottom) * SMOOTH_SPEED
        self.left += (self.t_left - self.left) * SMOOTH_SPEED

# ──────────────────────────────────────────────
# DATABASE MANAGER (Secure JSON Storage)
# ──────────────────────────────────────────────
class FaceDatabase:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.known_encodings = [np.array(enc) for enc in data.get("encodings", [])]
                self.known_names     = data.get("names", [])
                logger.info(f"Database dimuat: {len(self.known_names)} identitas terdaftar.")
            except Exception as e: 
                logger.error(f"Gagal memuat database: {e}")
        else:
            logger.warning("Database tidak ditemukan. Memulai database kosong baru.")

    def save(self):
        try:
            data = {"names": self.known_names, "encodings": [enc.tolist() for enc in self.known_encodings]}
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            logger.info("Perubahan database berhasil disimpan.")
        except Exception as e: 
            logger.error(f"Gagal menyimpan database: {e}")

    def add_face(self, name: str, encoding: np.ndarray):
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        self.save()

    def remove_face(self, name: str) -> int:
        indices = [i for i, n in enumerate(self.known_names) if n == name]
        for i in sorted(indices, reverse=True):
            self.known_encodings.pop(i)
            self.known_names.pop(i)
        if indices: self.save()
        return len(indices)

    def unique_names(self) -> list[str]: return sorted(set(self.known_names))
    def count(self) -> int: return len(self.known_names)

# ──────────────────────────────────────────────
# AI WORKER PROCESS (Multi-core, Bypass GIL)
# ──────────────────────────────────────────────
def ai_worker_process(task_queue: mp.Queue, result_queue: mp.Queue, db_path: str):
    logger.info("AI Worker Process berjalan di background core.")
    db = FaceDatabase(db_path)
    
    # PERBAIKAN: Gunakan desimal (float) agar kotak presisi di tengah wajah!
    scale = 1.0 / FRAME_SCALE 

    while True:
        task = task_queue.get()
        if task is None: 
            logger.info("AI Worker Process dihentikan secara aman.")
            break

        command, payload = task
        if command == "RELOAD_DB":
            logger.info("AI Worker menyinkronkan ulang data ke memori.")
            db.load()
            continue

        if command == "PROCESS":
            mode, rgb_small = payload
            locations = face_recognition.face_locations(rgb_small, model="hog")
            
            if mode == "recognize":
                results = []
                if locations:
                    encodings = face_recognition.face_encodings(rgb_small, locations)
                    for enc, (top, right, bottom, left) in zip(encodings, locations):
                        name, confidence = UNKNOWN_LABEL, 0.0
                        if db.known_encodings:
                            distances = face_recognition.face_distance(db.known_encodings, enc)
                            best_idx  = int(np.argmin(distances))
                            if distances[best_idx] <= TOLERANCE:
                                name = db.known_names[best_idx]
                                confidence = round((1 - distances[best_idx]) * 100, 1)
                        results.append((top * scale, right * scale, bottom * scale, left * scale, name, confidence))
                result_queue.put(("RECOGNIZE_RESULT", results))

            elif mode == "register":
                encodings = face_recognition.face_encodings(rgb_small, locations) if locations else []
                full_locs = [(t * scale, r * scale, b * scale, l * scale) for t, r, b, l in locations]
                result_queue.put(("REGISTER_RESULT", encodings, full_locs))

# ──────────────────────────────────────────────
# MAIN GUI APP (Main Thread)
# ──────────────────────────────────────────────
class FaceRecognitionApp(tk.Tk):
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.title("🎭 Enterprise Face Recognition System")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        logger.info("Menginisialisasi Antarmuka Pengguna...")
        self.db = FaceDatabase()
        self.task_queue = task_queue
        self.result_queue = result_queue
        
        self.running = True
        self.mode = "recognize" 
        self.current_frame = None
        self.animated_boxes: list[SmoothBox] = []
        
        self.reg_name = ""
        self.reg_encodings = []
        self.reg_target = 5
        self._last_sample = 0.0
        self._last_log_cache = {}

        self._build_ui()
        threading.Thread(target=self._camera_thread, daemon=True).start()
        threading.Thread(target=self._result_receiver, daemon=True).start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._ui_render_loop()

    def _build_ui(self):
        hdr = tk.Frame(self, bg="#16213e", pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎭 Secure Face Recognition", font=("Segoe UI", 16, "bold"), fg="#e94560", bg="#16213e").pack()

        self.canvas = tk.Canvas(self, width=640, height=480, bg="#000", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(6, 0))

        self.status_var = tk.StringVar(value="📷 Menghubungkan ke hardware kamera...")
        tk.Label(self, textvariable=self.status_var, font=("Segoe UI", 10), fg="#a8dadc", bg="#1a1a2e", anchor="w", padx=12).pack(fill="x", pady=(2, 0))

        ctrl = tk.Frame(self, bg="#1a1a2e", pady=8)
        ctrl.pack(fill="x", padx=10, pady=4)
        btn = {"font": ("Segoe UI", 10, "bold"), "relief": "flat", "cursor": "hand2", "padx": 14, "pady": 7, "bd": 0}

        tk.Button(ctrl, text="➕ Daftarkan Wajah", bg="#e94560", fg="white", command=self._start_register, **btn).pack(side="left", padx=4)
        tk.Button(ctrl, text="🗑 Hapus Wajah", bg="#533483", fg="white", command=self._delete_face, **btn).pack(side="left", padx=4)
        tk.Button(ctrl, text="📋 Daftar Wajah", bg="#0f3460", fg="white", command=self._show_list, **btn).pack(side="left", padx=4)
        tk.Button(ctrl, text="📄 Lihat Log", bg="#0f3460", fg="white", command=self._show_log, **btn).pack(side="left", padx=4)

        self.progress_frame = tk.Frame(self, bg="#1a1a2e")
        tk.Label(self.progress_frame, text="Progres Registrasi:", fg="#a8dadc", bg="#1a1a2e", font=("Segoe UI", 9)).pack(side="left", padx=(12, 6))
        self.progress = ttk.Progressbar(self.progress_frame, length=200, maximum=self.reg_target)
        self.progress.pack(side="left")
        self.lbl_prog = tk.Label(self.progress_frame, text="0/5", fg="#a8dadc", bg="#1a1a2e", font=("Segoe UI", 9))
        self.lbl_prog.pack(side="left", padx=6)
        tk.Button(self.progress_frame, text="✖ Batal", bg="#e94560", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", command=self._cancel_register).pack(side="left", padx=6)

    # ── CAMERA THREAD (I/O Latar Belakang) ────────
    def _camera_thread(self):
        logger.info("Mengakses feed video...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        self.after(0, lambda: self.status_var.set(f"✅ Sistem Stabil | Database: {self.db.count()} identitas"))

        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            self.current_frame = frame 
            
            try:
                small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                self.task_queue.put_nowait(("PROCESS", (self.mode, rgb_small)))
            except queue.Full: 
                pass # Buang frame jika AI Process masih sibuk (mencegah lag menumpuk)
                
        cap.release()
        logger.info("Kamera dinonaktifkan.")

    # ── RESULT RECEIVER THREAD ────────────────────
    def _result_receiver(self):
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.5)
                msg_type = result[0]
                
                if msg_type in ("RECOGNIZE_RESULT", "REGISTER_RESULT"):
                    ai_boxes = result[1] if msg_type == "RECOGNIZE_RESULT" else result[2]
                    
                    for item in ai_boxes:
                        if msg_type == "RECOGNIZE_RESULT":
                            t, r, b, l, name, conf = item
                        else:
                            t, r, b, l = item
                            name, conf = self.reg_name, 0.0

                        cx, cy = (l + r) / 2, (t + b) / 2
                        best_box = None
                        best_dist = float('inf')
                        
                        for box in self.animated_boxes:
                            bx, by = (box.left + box.right) / 2, (box.top + box.bottom) / 2
                            dist = math.hypot(cx - bx, cy - by)
                            if dist < TRACK_RADIUS and dist < best_dist:
                                best_dist = dist
                                best_box = box
                        
                        if best_box:
                            best_box.update_target(t, r, b, l, name, conf)
                        else:
                            self.animated_boxes.append(SmoothBox(t, r, b, l, name, conf))

                        if msg_type == "RECOGNIZE_RESULT" and name != UNKNOWN_LABEL:
                            self.after(0, self._write_log, name, conf)

                    # PERBAIKAN: Hanya catat frame registrasi JIKA state adalah "register" murni
                    if msg_type == "REGISTER_RESULT" and self.mode == "register":
                        encodings = result[1]
                        now = time.time()
                        if encodings and (now - self._last_sample) >= 0.6:
                            self.reg_encodings.append(encodings[0])
                            self._last_sample = now
                            self.after(0, self._update_reg_progress)

            except queue.Empty: continue

    # ── UI RENDER LOOP (60 FPS Polling) ───────────
    def _ui_render_loop(self):
        if not self.running: return

        frame = self.current_frame
        if frame is not None:
            display_frame = frame.copy()
            current_time = time.time()
            
            # Hapus kotak yang melewati masa grace period (wajah keluar frame)
            self.animated_boxes = [b for b in self.animated_boxes if (current_time - b.last_seen) < GRACE_PERIOD]

            for box in self.animated_boxes:
                box.glide() 
                
                # Konversi Float ke Int KHUSUS SAAT MENGGAMBAR, jangan di logic kalkulasi
                t, r, b, l = int(box.top), int(box.right), int(box.bottom), int(box.left)
                color = (0, 220, 100) if box.name != UNKNOWN_LABEL else (0, 80, 220)
                label = f"{box.name} ({box.conf}%)" if self.mode == "recognize" and box.name != UNKNOWN_LABEL else (
                        f"Registrasi: {self.reg_name}" if self.mode in ("register", "saving") else UNKNOWN_LABEL)
                
                cv2.rectangle(display_frame, (l, t), (r, b), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display_frame, (l, b), (l + tw + 8, b + th + 10), color, cv2.FILLED)
                cv2.putText(display_frame, label, (l + 4, b + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
            self.canvas.img_tk = img_tk

        self.after(UI_REFRESH_MS, self._ui_render_loop)

    # ── LOGIKA REGISTRASI ─────────────────────────
    def _start_register(self):
        name = simpledialog.askstring("Daftarkan Wajah", "Masukkan nama:", parent=self)
        if not name or not name.strip(): return
        
        self.reg_name, self.reg_encodings, self._last_sample = name.strip(), [], 0.0
        self.animated_boxes.clear()
        self.mode = "register"
        logger.info(f"==> MULAI REGISTRASI: {self.reg_name}")
        
        self.progress["value"] = 0
        self.lbl_prog.config(text=f"0/{self.reg_target}")
        self.progress_frame.pack(fill="x", padx=10, pady=(0, 6))
        self.status_var.set(f"📸 Mode registrasi aktif: {self.reg_name}")

    def _update_reg_progress(self):
        if self.mode != "register": return # PERBAIKAN BUG RACE CONDITION
            
        count = len(self.reg_encodings)
        self.progress["value"] = count
        self.lbl_prog.config(text=f"{count}/{self.reg_target}")
        logger.info(f"Proses Registrasi [{self.reg_name}]: {count}/{self.reg_target}")
        
        if count >= self.reg_target: 
            self.mode = "saving" # Kunci state pendaftaran
            self._finish_register()

    def _finish_register(self):
        logger.info(f"Kalkulasi rata-rata vektor untuk '{self.reg_name}'...")
        self.db.add_face(self.reg_name, np.mean(self.reg_encodings, axis=0))
        self.task_queue.put(("RELOAD_DB", None)) # Paksa proses AI update DB
        
        msg = self.reg_name
        self._cancel_register()
        logger.info(f"==> SUKSES: Wajah '{msg}' terdaftar.")
        messagebox.showinfo("Berhasil", f"✅ Data '{msg}' tersimpan.")

    def _cancel_register(self):
        if self.mode == "register": logger.warning("Proses registrasi dibatalkan.")
        self.mode = "recognize"
        self.animated_boxes.clear()
        self.progress_frame.pack_forget()
        self.status_var.set(f"✅ Sistem Stabil | Database: {self.db.count()} identitas")

    # ── UTILITAS LAINNYA ──────────────────────────
    def _delete_face(self):
        names = self.db.unique_names()
        if not names: return messagebox.showinfo("Info", "Database kosong.")
        win = tk.Toplevel(self)
        win.title("Hapus Data")
        win.configure(bg="#1a1a2e")
        lb = tk.Listbox(win, bg="#16213e", fg="white", selectbackground="#e94560")
        for n in names: lb.insert(tk.END, n)
        lb.pack(padx=20, pady=10)
        
        def do_delete():
            sel = lb.curselection()
            if not sel: return
            name = lb.get(sel[0])
            if messagebox.askyesno("Otorisasi Penghapusan", f"Yakin hapus data {name}?"):
                logger.info(f"Menghapus identitas: {name}")
                self.db.remove_face(name)
                self.task_queue.put(("RELOAD_DB", None))
                win.destroy()
                self.status_var.set(f"✅ Sistem Stabil | Database: {self.db.count()} identitas")
        tk.Button(win, text="Hapus Data", bg="#e94560", fg="white", command=do_delete).pack()

    def _show_list(self):
        names = self.db.unique_names()
        win = tk.Toplevel(self)
        win.title("Daftar Identitas Tervalidasi")
        win.configure(bg="#1a1a2e")
        lb = tk.Listbox(win, bg="#16213e", fg="white", width=40, height=15)
        for i, n in enumerate(names, 1): lb.insert(tk.END, f"{i:02d}. {n}")
        lb.pack(padx=20, pady=20)

    def _write_log(self, name: str, confidence: float):
        now = time.time()
        if now - self._last_log_cache.get(name, 0) < 5: return
        self._last_log_cache[name] = now
        
        # Log terminal realtime
        logger.info(f"MATCH: {name} (Akurasi: {confidence}%)")
        # Simpan audit trail ke text file
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Match: {name} ({confidence}%)\n")
        except: pass

    def _show_log(self):
        if not os.path.exists(LOG_FILE): return
        win = tk.Toplevel(self)
        win.title("System Log")
        txt = tk.Text(win, bg="#16213e", fg="#a8dadc", width=65, height=20)
        txt.pack(padx=10, pady=10)
        with open(LOG_FILE, "r") as f: txt.insert("1.0", f.read())
        txt.config(state="disabled")

    def _on_close(self):
        logger.info("Menutup sistem secara aman. Membersihkan resources...")
        self.running = False
        self.task_queue.put(None) 
        self.destroy()

# ──────────────────────────────────────────────
# BOOTSTRAP POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Wajib ada agar Multiprocessing berjalan stabil di OS Windows
    mp.freeze_support() 
    logger.info("=== ENTERPRISE FACE RECOGNITION BOOTING UP ===")
    
    # IPC (Inter-Process Communication) Queues
    task_queue = mp.Queue(maxsize=2) 
    result_queue = mp.Queue()
    
    # Memulai OS Process terpisah untuk memecah beban CPU
    ai_process = mp.Process(target=ai_worker_process, args=(task_queue, result_queue, DB_FILE))
    ai_process.daemon = True
    ai_process.start()
    
    # Memulai antarmuka GUI di Main Process
    app = FaceRecognitionApp(task_queue, result_queue)
    app.mainloop()