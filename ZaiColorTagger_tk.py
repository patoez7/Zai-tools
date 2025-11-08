import shutil
import os, re, csv, traceback, tempfile, time
from pathlib import Path
import threading
from typing import Dict, Any, List

# --- GUI ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- Audio ---
import numpy as np
import librosa
import soundfile as sf
import winsound  # Windows audition

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"}

# 14 spectral "color" bands (c1..c14)
COLOR_BANDS = [
    (20, 45), (45, 65), (65, 90), (90, 120),
    (120, 170), (170, 250), (250, 350), (350, 500),
    (500, 700), (700, 1000), (1000, 1600),
    (1600, 3000), (3000, 6000), (6000, 20000),
]

# filename tags
ZC_TAG_RE = re.compile(r"__Zc\d{1,2}\b", re.IGNORECASE)
ZBA_RE    = re.compile(r"__Zba\b", re.IGNORECASE)
ZSY_RE    = re.compile(r"__Zsy\b", re.IGNORECASE)

# ----------------- Low-level audio helpers -----------------

def load_mono(path, sr=48000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=40)
    return y, sr

def band_index_from_spectrum(y, sr):
    n_fft = 2048
    hop = n_fft // 4
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_energy = []
    for lo, hi in COLOR_BANDS:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        band_energy.append(float(S[idx, :].sum()) if len(idx) else 0.0)
    return int(np.argmax(band_energy))

def estimate_f0_features(y, sr):
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'), sr=sr)
    valid = f0[~np.isnan(f0)]
    if valid.size == 0:
        return None, 0.0, 0.0
    med = float(np.median(valid))
    cents = 1200 * np.log2(valid / med)
    stability = float(np.clip(1.0 - (np.nanstd(cents) / 100.0), 0.0, 1.0))
    return med, stability, valid.size / f0.size

def harmonic_percussive_ratio(y):
    S = librosa.stft(y)
    H, P = librosa.decompose.hpss(S)
    h = np.mean(np.abs(H)); p = np.mean(np.abs(P))
    return h / (h + p) if (h + p) > 0 else 0.5

def spectral_stats(y, sr):
    S = np.abs(librosa.stft(y)) + 1e-12
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    return centroid, flat

def is_bass_sound(f0_med, centroid):
    if f0_med and f0_med < 150: return True
    if centroid < 200: return True
    return False

def is_synth_sound(hpr, f0_stability, f0_valid_ratio, flatness):
    return (hpr > 0.60) and (f0_stability > 0.50) and (f0_valid_ratio > 0.35) and (flatness < 0.25)

def clean_tags(stem):
    stem = ZC_TAG_RE.sub("", stem)
    stem = ZBA_RE.sub("", stem)
    stem = ZSY_RE.sub("", stem)
    return stem

def ensure_unique_path(dst: Path):
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}({i}){ext}")
        if not cand.exists():
            return cand
        i += 1

# ----------------- Streaming analyzer -----------------

def analyze_file(path: Path, flags: Dict[str, bool]) -> Dict[str, Any]:
    """Analyze a single file according to selected flags."""
    y, sr = load_mono(str(path))
    if y.size == 0:
        raise RuntimeError("Empty/trimmed audio")

    # Always get something quick so row is useful immediately
    zc_idx = band_index_from_spectrum(y, sr) if flags.get("color", True) else None
    zc_str = f"Zc{zc_idx+1}" if zc_idx is not None else "-"

    # Optional/deeper features
    f0_med, f0_stab, f0_ratio = (None, 0.0, 0.0)
    hpr = None
    centroid = None
    flatness = None
    bass = synth = False

    if flags.get("f0", False) or flags.get("synth", False) or flags.get("bass", False):
        f0_med, f0_stab, f0_ratio = estimate_f0_features(y, sr)

    if flags.get("synth", False) or flags.get("bass", False):
        hpr = harmonic_percussive_ratio(y)
        centroid, flatness = spectral_stats(y, sr)

    if flags.get("bass", False):
        # uses f0 and/or centroid to decide
        bass = is_bass_sound(f0_med, centroid if centroid is not None else 9999)

    if flags.get("synth", False):
        synth = is_synth_sound(hpr if hpr is not None else 0.0,
                               f0_stab, f0_ratio,
                               flatness if flatness is not None else 1.0)

    # Build new name (only the tags you computed)
    stem = clean_tags(path.stem)
    tags = []
    if zc_idx is not None: tags.append(f"__Zc{zc_idx+1}")
    if flags.get("bass", False) and bass:  tags.append("__Zba")
    if flags.get("synth", False) and synth: tags.append("__Zsy")
    new_name = stem + "".join(tags) + path.suffix

    return {
        "full_path": str(path),
        "old_name": path.name,
        "new_name": new_name,
        "zc": zc_str,
        "bass": bool(bass),
        "synth": bool(synth),
    }

# ----------------- Rename / Undo -----------------

def apply_rename(rows, log_path, progress_cb=None, status_cb=None):
    changes = []
    total = len(rows)
    for i, row in enumerate(rows, 1):
        full_path = row["full_path"]; old_name = row["old_name"]; new_name = row["new_name"]
        p = Path(full_path)
        if new_name.startswith("ERROR"):
            continue
        if old_name == new_name:
            continue
        dst = p.with_name(new_name)
        dst = ensure_unique_path(dst)
        try:
            os.replace(p, dst)
            changes.append([str(p), str(dst)])
        except Exception as e:
            changes.append([str(p), f"ERROR: {e}"])
        if progress_cb:
            progress_cb(i, total)
    if changes:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["old_full_path", "new_full_path"])
            w.writerows(changes)
    if status_cb:
        ok = sum(1 for c in changes if not c[1].startswith("ERROR"))
        status_cb(f"Renamed {ok} files. Undo log saved at {log_path}")
    return changes

def undo_from_log(log_path, progress_cb=None, status_cb=None):
    if not Path(log_path).exists():
        if status_cb: status_cb("Log not found.")
        return []
    undone = []
    with open(log_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    total = len(rows)
    for i, row in enumerate(rows, 1):
        oldp = Path(row["old_full_path"])
        newp = Path(row["new_full_path"])
        if newp.exists():
            try:
                os.replace(newp, oldp)
                undone.append([str(newp), str(oldp)])
            except Exception as e:
                undone.append([str(newp), f"ERROR: {e}"])
        if progress_cb:
            progress_cb(i, total)
    if status_cb:
        status_cb(f"Undo complete: {len(undone)} items processed.")
    return undone

# ----------------- GUI -----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ZAI Color Tagger")
        self.geometry("1100x720")
        self.resizable(True, True)

        # Folder row
        frm = ttk.Frame(self); frm.pack(fill="x", padx=10, pady=10)
        ttk.Label(frm, text="Folder").pack(side="left")
        self.folder_var = tk.StringVar(value=r'D:\sample SLICES  code 0.1\New folder (giu app)')
        ttk.Entry(frm, textvariable=self.folder_var, width=82).pack(side="left", padx=6)
        ttk.Button(frm, text="Browse", command=self.pick_folder).pack(side="left", padx=3)

        # Analyze-for row
        af = ttk.LabelFrame(self, text="Analyze for (compute only selected)"); af.pack(fill="x", padx=10, pady=(0,6))
        self.flag_color = tk.BooleanVar(value=True)
        self.flag_bass  = tk.BooleanVar(value=True)
        self.flag_synth = tk.BooleanVar(value=True)
        self.flag_f0    = tk.BooleanVar(value=True)
        ttk.Checkbutton(af, text="Color (Zc1..Zc14)", variable=self.flag_color).pack(side="left", padx=6)
        ttk.Checkbutton(af, text="Bass tag (Zba)", variable=self.flag_bass).pack(side="left", padx=6)
        ttk.Checkbutton(af, text="Synth tag (Zsy)", variable=self.flag_synth).pack(side="left", padx=6)
        ttk.Checkbutton(af, text="Pitch features (f0)", variable=self.flag_f0).pack(side="left", padx=6)

        # Buttons row
        btns = ttk.Frame(self); btns.pack(fill="x", padx=10)
        self.btn_start = ttk.Button(btns, text="Analyze â–¶", command=self.on_start)
        self.btn_pause = ttk.Button(btns, text="Pause â¸", command=self.on_pause, state="disabled")
        self.btn_resume= ttk.Button(btns, text="Resume â–¶", command=self.on_resume, state="disabled")
        self.btn_stop  = ttk.Button(btns, text="Stop â¹", command=self.on_stop, state="disabled")

        self.btn_start.pack(side="left")
        self.btn_pause.pack(side="left", padx=6)
        self.btn_resume.pack(side="left")
        self.btn_stop.pack(side="left", padx=6)

        ttk.Button(btns, text="Apply Rename", command=self.on_apply).pack(side="left", padx=16)
        ttk.Button(btns, text="Undo Last", command=self.on_undo).pack(side="left")

        ttk.Button(btns, text="Filtersâ€¦", command=self.open_filters).pack(side="right")
        
        ttk.Button(btns, text="Show All (Analyzed)", command=self.show_all_analyzed).pack(side="right", padx=6)

        # Play controls
        playbar = ttk.Frame(self); playbar.pack(fill="x", padx=10, pady=(6,0))
        ttk.Button(playbar, text="â–¶ Play Selected", command=self.play_selected).pack(side="left")
        ttk.Button(playbar, text="â­ Play Next", command=self.play_next).pack(side="left", padx=6)
        ttk.Button(playbar, text="â¹ Stop", command=self.stop_playback).pack(side="left", padx=6)

        # Status + progress
        stat = ttk.Frame(self); stat.pack(fill="x", padx=10, pady=(6, 4))
        ttk.Label(stat, text="Log:").pack(side="left")
        self.status_var = tk.StringVar(value="")
        ttk.Label(stat, textvariable=self.status_var).pack(side="left", padx=6)
        self.pb = ttk.Progressbar(self, mode="determinate")
        self.pb.pack(fill="x", padx=10, pady=(0,8))

        # Table
        cols = ("Full Path","Old Name","New Name","Zc","Bass","Synth")
        self.table = ttk.Treeview(self, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, anchor="w", width=180 if c!="Full Path" else 520)
        self.table.pack(fill="both", expand=True, padx=10, pady=6)
        self.table.bind("<Double-1>", lambda e: self.play_selected())

        # State
        self.rows_all: List[Dict[str, Any]] = []
        self.rows_view: List[Dict[str, Any]] = []
        self.log_path = ""
        self.filters = {"mode": "all", "bass": False, "synth": False, "zc": ["Zc4","Zc7"]}
        self.filters_active = False
        self._tmp_wav = None
        self._worker = None
        self._pause_ev = threading.Event()
        self._stop_ev = threading.Event()
        self._last_flush = 0.0
        self._play_index = None  # for Play Next

    # ---------- helpers ----------
    def pick_folder(self):
        d = filedialog.askdirectory()
        if d: self.folder_var.set(d)

    def set_status(self, text):
        self.status_var.set(text)
        self.update_idletasks()

    def set_progress(self, i, total):
        self.pb["maximum"] = total if total else 1
        self.pb["value"] = i
        self.update_idletasks()

    def flags(self) -> Dict[str,bool]:
        # synth/bass depend on f0/centroid internally; handled in analyze_file
        return {
            "color": self.flag_color.get(),
            "bass":  self.flag_bass.get(),
            "synth": self.flag_synth.get(),
            "f0":    self.flag_f0.get(),
        }

    # ---------- streaming analyze controls ----------
    def on_start(self):
        folder = self.folder_var.get().strip()
        if not folder or not Path(folder).exists():
            messagebox.showerror("Error", "Pick a valid folder.")
            return
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Running", "Analysis already running.")
            return

        self.rows_all.clear()
        self.rows_view.clear()
        self.table.delete(*self.table.get_children())
        self._stop_ev.clear()
        self._pause_ev.clear()
        self.pb["value"] = 0
        self.set_status("Analyzing (streaming)â€¦")

        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.btn_resume.config(state="disabled")

        self.log_path = str(Path(folder) / ("rename_log_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"))
        self._worker = threading.Thread(target=self._worker_loop, args=(folder, self.flags()), daemon=True)
        self._worker.start()

    def _worker_loop(self, folder: str, flags: Dict[str,bool]):
        files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        total = len(files)
        self._last_flush = time.time()
        pending_for_ui = []
        added_since_flush = 0

        for i, p in enumerate(files, 1):
            # Pause loop
            while self._pause_ev.is_set() and not self._stop_ev.is_set():
                time.sleep(0.05)

            if self._stop_ev.is_set():
                break

            try:
                row = analyze_file(Path(p), flags)
            except Exception as e:
                row = {
                    "full_path": str(p),
                    "old_name": Path(p).name,
                    "new_name": f"ERROR: {e}",
                    "zc": "-",
                    "bass": False,
                    "synth": False
                }

            # store and schedule UI insert (respect filters while streaming)
            self.rows_all.append(row)

            if (not self.filters_active) or self.match_row(row):
                self.rows_view.append(row)
                pending_for_ui.append(row)
                added_since_flush += 1

            # rate-limited flush (every 0.5s or 10 visible items, or at end)
            now = time.time()
            if pending_for_ui and ((now - self._last_flush) >= 0.5 or added_since_flush >= 10 or i == total):
                batch = pending_for_ui[:]
                pending_for_ui.clear()
                self.after(0, self._insert_rows_batch, batch)
                self._last_flush = now
                added_since_flush = 0

            # progress bar
            self.after(0, self.set_progress, i, total)

        # finished / stopped
        if not self._stop_ev.is_set():
            self.after(0, self.set_status, f"Preview ready. {len(self.rows_all)} files. Log will be {self.log_path}")
        else:
            self.after(0, self.set_status, f"Stopped at {len(self.rows_all)} items.")

        self.after(0, self._set_idle_buttons)

    

    def _insert_rows_batch(self, rows: List[Dict[str,Any]]):
        for r in rows:
            self.table.insert("", "end", values=[
                r["full_path"], r["old_name"], r["new_name"],
                r["zc"], "Yes" if r["bass"] else "No", "Yes" if r["synth"] else "No"
            ])

    def _set_idle_buttons(self):
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.btn_resume.config(state="disabled")

    def on_pause(self):
        if self._worker and self._worker.is_alive():
            self._pause_ev.set()
            self.btn_pause.config(state="disabled")
            self.btn_resume.config(state="normal")
            self.set_status("Paused.")

    def on_resume(self):
        if self._worker and self._worker.is_alive():
            self._pause_ev.clear()
            self.btn_pause.config(state="normal")
            self.btn_resume.config(state="disabled")
            self.set_status("Resumed.")

    def on_stop(self):
        if self._worker and self._worker.is_alive():
            self._stop_ev.set()
            self._pause_ev.clear()
            self.set_status("Stoppingâ€¦")
            # buttons set back when thread exits

    # ---------- Filters ----------
    def open_filters(self):
        FilterDialog(self, self.filters, self.set_filters)
    def set_filters(self, cfg):
        self.filters = cfg
        self.filters_active = True           # filters ON
        self.apply_filters()                 # apply immediately
        self.set_status("Filters applied.")

    def apply_filters(self):
        if not self.rows_all:
            # allow applying before analysis finishes; table just clears or shows none
            self.rows_view = []
            self.table.delete(*self.table.get_children())
            return

        # rebuild visible table using match_row()
        self.rows_view = [r for r in self.rows_all if self.match_row(r)]
        self.table.delete(*self.table.get_children())
        self._insert_rows_batch(self.rows_view)
        self.set_status(f"Applied filters: {len(self.rows_view)} shown.")

    def match_row(self, row):
        mode = self.filters.get("mode", "any")
        want_bass  = self.filters.get("bass", False)
        want_synth = self.filters.get("synth", False)
        want_zc    = set(self.filters.get("zc", []))  # e.g., {"Zc4","Zc7"}

        # If nothing selected, show everything.
        if not (want_bass or want_synth or want_zc):
            return True

        tags = set()
        if row.get("bass"):   tags.add("Zba")
        if row.get("synth"):  tags.add("Zsy")
        if row.get("zc"):     tags.add(row["zc"])     # e.g., "Zc4"

        selected = set()
        if want_bass:  selected.add("Zba")
        if want_synth: selected.add("Zsy")
        selected |= want_zc

        if mode == "all":
            return selected.issubset(tags)
        else:
            return not selected.isdisjoint(tags)
    def show_all_analyzed(self):
        # Turn filters off and repopulate the visible list from everything analyzed so far
        self.filters_active = False
        self.table.delete(*self.table.get_children())
        self.rows_view = list(self.rows_all)
        if self.rows_view:
            batch = 200
            for i in range(0, len(self.rows_view), batch):
                self._insert_rows_batch(self.rows_view[i:i+batch])
                self.update_idletasks()
            self.set_status(f"Showing all analyzed: {len(self.rows_view)}")
        else:
            self.set_status("No analyzed files yet.")


    def show_all_analyzed(self):
        # Turn filters off and repopulate the visible list from everything analyzed so far
        self.filters_active = False
        self.table.delete(*self.table.get_children())
        self.rows_view = list(self.rows_all)
        if self.rows_view:
            batch = 200
            for i in range(0, len(self.rows_view), batch):
                self._insert_rows_batch(self.rows_view[i:i+batch])
                self.update_idletasks()
            self.set_status(f"Showing all analyzed: {len(self.rows_view)}")
        else:
            self.set_status("No analyzed files yet.")

    # ---------- Rename / Undo ----------


    def on_apply(self):
        if not self.rows_view:
            self.set_status("Nothing to rename. Apply filters or Analyze first.")
            return
        self.set_status("Renamingâ€¦")
        self.pb["value"] = 0

        def work():
            try:
                apply_rename(
                    self.rows_view,
                    self.log_path or str(Path(self.folder_var.get())/"rename_log.csv"),
                    progress_cb=self.set_progress,
                    status_cb=self.set_status
                )
            except Exception as e:
                self.set_status(f"Error: {e}")
                traceback.print_exc()

        threading.Thread(target=work, daemon=True).start()

    def on_undo(self):
        folder = self.folder_var.get().strip()
        if not folder:
            self.set_status("Pick a folder first.")
            return
        logp = self.log_path or str(Path(folder) / "rename_log.csv")
        if not Path(logp).exists():
            messagebox.showwarning("Undo", "rename_log.csv not found in the folder.")
            return
        self.set_status("Undoingâ€¦")
        self.pb["value"] = 0

        def work():
            try:
                undo_from_log(
                    logp,
                    progress_cb=self.set_progress,
                    status_cb=self.set_status
                )
            except Exception as e:
                self.set_status(f"Error: {e}")
                traceback.print_exc()

        threading.Thread(target=work, daemon=True).start()

    # ---------- Playback ----------
    def play_selected(self):
        sel = self.table.selection()
        if not sel:
            self.set_status("Select a row to play.")
            return
        idx = self.table.index(sel[0])
        self._play_index = idx  # set playhead for Play Next
        self._play_row(sel[0])

    def play_next(self):
        # Decide next item based on current selection / index
        children = self.table.get_children()
        if not children:
            self.set_status("No items to play.")
            return

        if self._play_index is None:
            # start from first
            self._play_index = 0
        else:
            self._play_index = min(self._play_index + 1, len(children) - 1)

        iid = children[self._play_index]
        self.table.selection_set(iid)
        self.table.see(iid)
        self._play_row(iid)

    def _play_row(self, iid):
        vals = self.table.item(iid, "values")
        path = vals[0]
        p = Path(path)
        if not p.exists():
            self.set_status("File not found on disk.")
            return

        self.stop_playback()

        try:
            if p.suffix.lower() == ".wav":
                winsound.PlaySound(str(p), winsound.SND_ASYNC | winsound.SND_FILENAME)
                self.set_status(f"Playing: {p.name}")
            else:
                y, sr = load_mono(str(p), sr=48000)
                tmp = Path(tempfile.gettempdir()) / ("zai_preview.wav")
                sf.write(str(tmp), y, 48000)
                self._tmp_wav = str(tmp)
                winsound.PlaySound(self._tmp_wav, winsound.SND_ASYNC | winsound.SND_FILENAME)
                self.set_status(f"Playing (converted): {p.name}")
        except Exception as e:
            self.set_status(f"Preview error: {e}")

    def stop_playback(self):
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

# ----------------- Filter dialog -----------------

class FilterDialog(tk.Toplevel):
    def __init__(self, master, current, on_apply):
        super().__init__(master)
        self.title("Filters")
        self.resizable(False, False)
        self.on_apply = on_apply

        frm = ttk.Frame(self, padding=10); frm.pack(fill="both", expand=True)

        # Mode
        self.mode = tk.StringVar(value=current.get("mode","any"))
        ttk.Label(frm, text="Mode:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frm, text="Individual (OR)", variable=self.mode, value="any").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frm, text="Specific (AND)", variable=self.mode, value="all").grid(row=0, column=2, sticky="w")

        # Tags
        self.bass = tk.BooleanVar(value=current.get("bass", False))
        self.synth = tk.BooleanVar(value=current.get("synth", False))
        ttk.Checkbutton(frm, text="Zba (Bass)", variable=self.bass).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(frm, text="Zsy (Synth)", variable=self.synth).grid(row=1, column=1, sticky="w")

        ttk.Separator(frm).grid(row=2, column=0, columnspan=3, pady=8, sticky="ew")

        ttk.Label(frm, text="Zc bands:").grid(row=3, column=0, sticky="w")

        self.zc_vars = {}
        zc_frame = ttk.Frame(frm); zc_frame.grid(row=4, column=0, columnspan=3, sticky="w")
        for i in range(14):
            var = tk.BooleanVar(value=(f"Zc{i+1}" in (current.get("zc", []) or ["Zc4","Zc7"])))
            self.zc_vars[f"Zc{i+1}"] = var
            col = i % 7
            row = i // 7
            ttk.Checkbutton(zc_frame, text=f"Zc{i+1}", variable=var).grid(row=row, column=col, padx=4, pady=2, sticky="w")

        btns = ttk.Frame(frm); btns.grid(row=5, column=0, columnspan=3, pady=(10,0), sticky="ew")
        ttk.Button(btns, text="Apply", command=self.apply).pack(side="left")
        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")

        self.grab_set()
        self.transient(master)

    def apply(self):
        chosen_zc = [k for k, v in self.zc_vars.items() if v.get()]
        cfg = {
            "mode": self.mode.get(),      # "any" or "all"
            "bass": self.bass.get(),
            "synth": self.synth.get(),
            "zc": chosen_zc
        }
        self.on_apply(cfg)

# ----------------- main -----------------

if __name__ == "__main__":
    App().mainloop()












# === ZAI FEATURE PACK START ===
# Utilities added non-destructively so we don't break your existing code.

def zai_extract_song_basename(filename: str):
    # Pull base song name (before last _### / -### / space###)
    name = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^(.*?)[ _-]?(\d+)$', name)
    return (m.group(1) if m else name).strip()

def zai_scan_folder_for_chops(folder):
    exts = {'.wav','.aif','.aiff','.flac','.mp3','.ogg','.m4a'}
    rows = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                full = os.path.join(root, f)
                base = zai_extract_song_basename(f)
                # guess chop number
                num = None
                m = re.match(r'.*?([0-9]+)(?!.*[0-9])', os.path.splitext(f)[0])
                if m: 
                    try: num = int(m.group(1))
                    except: num = None
                rows.append({
                    "file": full,
                    "song": base,
                    "chop_no": num,
                    # placeholders—fill from your analysis if available:
                    "zc": None, "bass": None, "synth": None, "f0": None, "loud": None
                })
    return rows

def zai_tree_make_sortable(tree):
    # Click any header to sort asc/desc. Numeric if possible.
    def _to_num(s):
        try:
            return float(s)
        except:
            return s
    def _sort(col, reverse):
        data = [(tree.set(k, col), k) for k in tree.get_children('')]
        # decide numeric?
        try:
            [_to_num(v) for v,_ in data]
            numeric = all(isinstance(_to_num(v), (int,float)) for v,_ in data)
        except:
            numeric = False
        keyfunc = (lambda t: _to_num(t[0])) if numeric else (lambda t: t[0])
        data.sort(key=keyfunc, reverse=reverse)
        for idx, (_, k) in enumerate(data):
            tree.move(k, '', idx)
        tree.heading(col, command=lambda: _sort(col, not reverse))
    for c in tree["columns"]:
        tree.heading(c, command=lambda c=c: _sort(c, False))

def zai_add_context_menu(root, tree):
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="Copy paths (Ctrl+C)", command=lambda: root.event_generate("<<CopySelectedPaths>>"))
    menu.add_command(label="Open containing folder", command=lambda: root.event_generate("<<OpenContaining>>"))
    menu.add_command(label="Export selected…", command=lambda: root.event_generate("<<ExportSelected>>"))
    def _menu(e):
        iid = tree.identify_row(e.y)
        if iid:
            tree.selection_set(iid)
            menu.tk_popup(e.x_root, e.y_root)
    tree.bind("<Button-3>", _menu)  # right-click

def zai_bind_copy_open_export(root, tree):
    def get_selected_files():
        paths = []
        for iid in tree.selection():
            p = tree.set(iid, "file")
            if p: paths.append(p)
        return paths

    def on_copy(event=None):
        try:
            paths = get_selected_files()
            root.clipboard_clear()
            root.clipboard_append("\n".join(paths))
        except Exception as e:
            messagebox.showerror("Copy", str(e))

    def on_open(event=None):
        import subprocess, os
        paths = get_selected_files()
        if not paths: return
        # open first's folder
        folder = os.path.dirname(paths[0])
        subprocess.Popen(f'explorer.exe "{folder}"')

    def on_export(event=None):
        from tkinter import filedialog
        dest = filedialog.askdirectory(title="Export selected to folder")
        if not dest: return
        for p in get_selected_files():
            try:
                shutil.copy2(p, dest)
            except Exception as e:
                messagebox.showwarning("Export", f"Couldn't export {os.path.basename(p)}: {e}")
        messagebox.showinfo("Export", "Done!")

    root.bind("<<CopySelectedPaths>>", on_copy)
    root.bind("<<OpenContaining>>", on_open)
    root.bind("<<ExportSelected>>", on_export)
    root.bind("<Control-c>", on_copy)

def zai_save_csv(folder, table_rows):
    # persist analysis/index info to CSV in working folder
    import csv
    out = os.path.join(folder, "analysis_index.csv")
    cols = ["file","song","chop_no","zc","bass","synth","f0","loud"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table_rows:
            w.writerow({k: r.get(k) for k in cols})
    return out

def zai_confirm_then(func):
    def wrapper(*a, **kw):
        if messagebox.askyesno("Party", "Do you wanna party?"):
            return func(*a, **kw)
    return wrapper
# === ZAI FEATURE PACK END ===





