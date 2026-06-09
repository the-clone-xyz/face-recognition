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
AI_MIN_INTERVAL = 0.06   # Batasi beban AI, tetapi tetap responsif
AI_CONFIRM_GRACE = 2.2   # Kotak tetap hidup sebentar saat AI miss sesaat
FLOW_MAX_POINTS = 80
FLOW_MIN_POINTS = 8
FLOW_RESEED_INTERVAL = 0.45
FLOW_MAX_STEP = 45.0

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
        self.last_ai_seen = self.last_seen
        self.flow_points = None
        self.last_flow_seed = 0.0

    def update_target(self, top, right, bottom, left, name, conf):
        self.t_top, self.t_right, self.t_bottom, self.t_left = top, right, bottom, left
        if name != UNKNOWN_LABEL or self.name == UNKNOWN_LABEL:
            self.name = name
            self.conf = conf
        now = time.time()
        self.last_seen = now
        self.last_ai_seen = now
        self.flow_points = None
        self.last_flow_seed = 0.0

    def apply_motion(self, dx, dy):
        self.left += dx
        self.right += dx
        self.t_left += dx
        self.t_right += dx
        self.top += dy
        self.bottom += dy
        self.t_top += dy
        self.t_bottom += dy
        self.last_seen = time.time()

    def glide(self):
        # Adaptive lerp: tetap halus untuk gerakan kecil, cepat mengejar gerakan besar.
        gap = max(
            abs(self.t_top - self.top),
            abs(self.t_right - self.right),
            abs(self.t_bottom - self.bottom),
            abs(self.t_left - self.left),
        )
        speed = min(0.72, max(SMOOTH_SPEED, gap / 180.0))
        self.top += (self.t_top - self.top) * speed
        self.right += (self.t_right - self.right) * speed
        self.bottom += (self.t_bottom - self.bottom) * speed
        self.left += (self.t_left - self.left) * speed

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
            if len(payload) == 3:
                seq, mode, rgb_small = payload
            else:
                seq = 0
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
                result_queue.put(("RECOGNIZE_RESULT", seq, mode, results))

            elif mode == "register":
                encodings = face_recognition.face_encodings(rgb_small, locations) if locations else []
                full_locs = [(t * scale, r * scale, b * scale, l * scale) for t, r, b, l in locations]
                result_queue.put(("REGISTER_RESULT", seq, mode, encodings, full_locs))

            else:
                result_queue.put(("PROCESS_SKIPPED", seq, mode))

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
        self.box_lock = threading.Lock()
        self.ai_lock = threading.Lock()
        self._ai_inflight = False
        self._ai_inflight_seq = 0
        self._ai_seq = 0
        self._latest_result_seq = 0
        self._last_ai_submit = 0.0
        self._prev_gray = None
        
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
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        self.after(0, lambda: self.status_var.set(f"✅ Sistem Stabil | Database: {self.db.count()} identitas"))

        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            with self.box_lock:
                if self._prev_gray is not None and self.animated_boxes:
                    self._update_motion_tracking(self._prev_gray, gray)
                self._prev_gray = gray

            self.current_frame = frame 
            self._submit_frame_for_ai(frame)
                
        cap.release()
        logger.info("Kamera dinonaktifkan.")

    def _submit_frame_for_ai(self, frame):
        mode = self.mode
        if mode not in ("recognize", "register"):
            return

        now = time.time()
        with self.ai_lock:
            if self._ai_inflight or (now - self._last_ai_submit) < AI_MIN_INTERVAL:
                return
            self._ai_seq += 1
            seq = self._ai_seq
            self._ai_inflight = True
            self._ai_inflight_seq = seq
            self._last_ai_submit = now

        try:
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            self.task_queue.put_nowait(("PROCESS", (seq, mode, rgb_small)))
        except queue.Full:
            with self.ai_lock:
                if seq == self._ai_inflight_seq:
                    self._ai_inflight = False
        except Exception as e:
            logger.error(f"Gagal mengirim frame ke AI Worker: {e}")
            with self.ai_lock:
                if seq == self._ai_inflight_seq:
                    self._ai_inflight = False

    def _accept_ai_result(self, seq):
        with self.ai_lock:
            if seq < self._latest_result_seq:
                return False
            self._latest_result_seq = seq
            if seq >= self._ai_inflight_seq:
                self._ai_inflight = False
            return True

    def _seed_flow_points(self, gray, box: SmoothBox):
        height, width = gray.shape[:2]
        left = max(0, min(width - 1, int(box.t_left)))
        right = max(0, min(width, int(box.t_right)))
        top = max(0, min(height - 1, int(box.t_top)))
        bottom = max(0, min(height, int(box.t_bottom)))

        if (right - left) < 20 or (bottom - top) < 20:
            box.flow_points = None
            return

        roi = gray[top:bottom, left:right]
        points = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=FLOW_MAX_POINTS,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
        )

        if points is None:
            box.flow_points = None
            return

        points[:, 0, 0] += left
        points[:, 0, 1] += top
        box.flow_points = points.astype(np.float32)
        box.last_flow_seed = time.time()

    def _update_motion_tracking(self, prev_gray, gray):
        now = time.time()
        height, width = gray.shape[:2]

        for box in self.animated_boxes:
            if (now - box.last_ai_seen) > AI_CONFIRM_GRACE:
                continue

            needs_seed = (
                box.flow_points is None
                or len(box.flow_points) < FLOW_MIN_POINTS
                or (now - box.last_flow_seed) > FLOW_RESEED_INTERVAL
            )
            if needs_seed:
                self._seed_flow_points(prev_gray, box)

            if box.flow_points is None or len(box.flow_points) < FLOW_MIN_POINTS:
                continue

            try:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray,
                    gray,
                    box.flow_points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
            except cv2.error:
                box.flow_points = None
                continue

            if next_points is None or status is None:
                box.flow_points = None
                continue

            valid = status.reshape(-1) == 1
            old_points = box.flow_points[valid].reshape(-1, 2)
            new_points = next_points[valid].reshape(-1, 2)

            if len(new_points) < FLOW_MIN_POINTS:
                box.flow_points = None
                continue

            in_frame = (
                (new_points[:, 0] >= 0)
                & (new_points[:, 0] < width)
                & (new_points[:, 1] >= 0)
                & (new_points[:, 1] < height)
            )
            old_points = old_points[in_frame]
            new_points = new_points[in_frame]

            if len(new_points) < FLOW_MIN_POINTS:
                box.flow_points = None
                continue

            deltas = new_points - old_points
            median = np.median(deltas, axis=0)
            spread = np.linalg.norm(deltas - median, axis=1)
            if len(spread) >= FLOW_MIN_POINTS:
                keep = spread <= max(6.0, np.percentile(spread, 75) * 2.0)
                if np.count_nonzero(keep) >= FLOW_MIN_POINTS:
                    deltas = deltas[keep]
                    new_points = new_points[keep]
                    median = np.median(deltas, axis=0)

            dx, dy = float(median[0]), float(median[1])
            if abs(dx) > FLOW_MAX_STEP or abs(dy) > FLOW_MAX_STEP:
                box.flow_points = None
                continue

            box.apply_motion(dx, dy)
            box.flow_points = new_points.reshape(-1, 1, 2).astype(np.float32)

    # ── RESULT RECEIVER THREAD ────────────────────
    def _result_receiver(self):
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.5)
                msg_type = result[0]

                if msg_type == "PROCESS_SKIPPED":
                    self._accept_ai_result(result[1])
                    continue

                if msg_type in ("RECOGNIZE_RESULT", "REGISTER_RESULT"):
                    seq = result[1]
                    result_mode = result[2]
                    if not self._accept_ai_result(seq):
                        continue

                    # Abaikan hasil lama dari mode sebelum user pindah ke register/cancel.
                    if result_mode != self.mode:
                        continue

                    ai_boxes = result[3] if msg_type == "RECOGNIZE_RESULT" else result[4]
                    log_events = []

                    with self.box_lock:
                        matched_indices = set()

                        for item in ai_boxes:
                            if msg_type == "RECOGNIZE_RESULT":
                                t, r, b, l, name, conf = item
                            else:
                                t, r, b, l = item
                                name, conf = self.reg_name, 0.0

                            cx, cy = (l + r) / 2, (t + b) / 2
                            best_idx = None
                            best_dist = float('inf')

                            for idx, box in enumerate(self.animated_boxes):
                                if idx in matched_indices:
                                    continue
                                bx, by = (box.left + box.right) / 2, (box.top + box.bottom) / 2
                                dist = math.hypot(cx - bx, cy - by)
                                if dist < TRACK_RADIUS and dist < best_dist:
                                    best_dist = dist
                                    best_idx = idx

                            if best_idx is not None:
                                self.animated_boxes[best_idx].update_target(t, r, b, l, name, conf)
                                matched_indices.add(best_idx)
                            else:
                                self.animated_boxes.append(SmoothBox(t, r, b, l, name, conf))

                            if msg_type == "RECOGNIZE_RESULT" and name != UNKNOWN_LABEL:
                                log_events.append((name, conf))

                    for name, conf in log_events:
                        self.after(0, self._write_log, name, conf)

                    # PERBAIKAN: Hanya catat frame registrasi JIKA state adalah "register" murni
                    if msg_type == "REGISTER_RESULT" and self.mode == "register":
                        encodings = result[3]
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

            boxes_to_draw = []
            with self.box_lock:
                # Kotak tetap hidup jika optical flow masih valid, tetapi wajib dikonfirmasi AI berkala.
                self.animated_boxes = [
                    b for b in self.animated_boxes
                    if (current_time - b.last_seen) < GRACE_PERIOD
                    and (current_time - b.last_ai_seen) < AI_CONFIRM_GRACE
                ]

                for box in self.animated_boxes:
                    box.glide() 

                    # Konversi Float ke Int KHUSUS SAAT MENGGAMBAR, jangan di logic kalkulasi
                    t, r, b, l = int(box.top), int(box.right), int(box.bottom), int(box.left)
                    boxes_to_draw.append((t, r, b, l, box.name, box.conf))

            for t, r, b, l, name, conf in boxes_to_draw:
                color = (0, 220, 100) if name != UNKNOWN_LABEL else (0, 80, 220)
                label = f"{name} ({conf}%)" if self.mode == "recognize" and name != UNKNOWN_LABEL else (
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
        with self.box_lock:
            self.animated_boxes.clear()
            self._prev_gray = None
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
        with self.box_lock:
            self.animated_boxes.clear()
            self._prev_gray = None
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
        try:
            self.task_queue.put_nowait(None)
        except queue.Full:
            pass
        self.destroy()

# ──────────────────────────────────────────────
# BOOTSTRAP POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Wajib ada agar Multiprocessing berjalan stabil di OS Windows
    mp.freeze_support() 
    logger.info("=== ENTERPRISE FACE RECOGNITION BOOTING UP ===")
    
    # IPC (Inter-Process Communication) Queues
    task_queue = mp.Queue(maxsize=1) 
    result_queue = mp.Queue()
    
    # Memulai OS Process terpisah untuk memecah beban CPU
    ai_process = mp.Process(target=ai_worker_process, args=(task_queue, result_queue, DB_FILE))
    ai_process.daemon = True
    ai_process.start()
    
    # Memulai antarmuka GUI di Main Process
    app = FaceRecognitionApp(task_queue, result_queue)
    app.mainloop()
