import argparse, os, math, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

AUDIO_EXTS = (".mp3",".wav",".flac",".ogg",".m4a",".aiff",".aif")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_audio(path: Path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y, sr, librosa.get_duration(y=y, sr=sr)

def write_wav(path: Path, y, sr):
    ensure_dir(path.parent)
    sf.write(str(path), y, sr)

def detect_onsets(y, sr, backtrack=True):
    # strong transient picks
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=backtrack)
    if on.size == 0:
        on = np.array([0.0])
    return on

def slice_by_onsets(y, sr, onsets_time, min_gap=0.08, max_len_s=None):
    # dedupe / spacing
    times = [0.0]
    for t in onsets_time:
        if t - times[-1] >= float(min_gap):
            times.append(float(t))
    dur = librosa.get_duration(y=y, sr=sr)
    times.append(dur)

    # build start/end samples
    segs = []
    for i in range(len(times)-1):
        s = int(times[i]*sr)
        e = int(times[i+1]*sr)
        if e - s <= 0: 
            continue
        if max_len_s is not None:
            e = min(e, s + int(max_len_s*sr))
        segs.append((s,e))
    return segs

def features(y, sr):
    # length
    dur = len(y)/sr
    # loudness-ish
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    # brightness
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    # rough pitch (Hz) – robust for percussive may be NaN
    try:
        f0 = librosa.yin(y, fmin=30, fmax=2000, sr=sr)
        f0 = float(np.nanmedian(f0))
    except Exception:
        f0 = float("nan")
    return {
        "duration_s": dur,
        "rms": rms,
        "peak": peak,
        "centroid": centroid,
        "rolloff": rolloff,
        "bandwidth": bandwidth,
        "zcr": zcr,
        "f0_hz": f0,
    }

def should_keep(m, spec):
    # spec holds thresholds; any missing is ignored
    def between(val, lo, hi):
        if np.isnan(val): 
            return False
        return (lo is None or val >= lo) and (hi is None or val <= hi)

    # duration
    if not between(m["duration_s"], spec.get("min_dur"), spec.get("max_dur")):
        return False
    # loudness surrogate
    if not between(m["rms"], spec.get("min_rms"), spec.get("max_rms")):
        return False
    # brightness
    if not between(m["centroid"], spec.get("min_centroid"), spec.get("max_centroid")):
        return False
    # pitch
    f0min, f0max = spec.get("min_f0"), spec.get("max_f0")
    if f0min is not None or f0max is not None:
        if np.isnan(m["f0_hz"]): 
            return False
        if not between(m["f0_hz"], f0min, f0max):
            return False
    return True

PRESETS = {
    # tweak these numbers to taste (centroid/rolloff are in Hz)
    "kicks": {
        "min_dur": 0.03, "max_dur": 0.25,
        "min_rms": 0.01, "max_rms": None,
        "min_centroid": None, "max_centroid": 2500,
        "min_f0": 35, "max_f0": 120,
    },
    "snares": {
        "min_dur": 0.05, "max_dur": 0.45,
        "min_rms": 0.01, "max_rms": None,
        "min_centroid": 1500, "max_centroid": 6000,
        "min_f0": None, "max_f0": None,
    },
    "hats": {
        "min_dur": 0.015, "max_dur": 0.18,
        "min_rms": 0.003, "max_rms": None,
        "min_centroid": 4500, "max_centroid": None,
        "min_f0": None, "max_f0": None,
    },
    "loops": {
        "min_dur": 0.4, "max_dur": 3.0,
        "min_rms": 0.005, "max_rms": None,
        "min_centroid": None, "max_centroid": None,
        "min_f0": None, "max_f0": None,
    },
}

def main():
    ap = argparse.ArgumentParser(description="Auto-chop transients, analyze slices, filter by specs.")
    ap.add_argument("--input", required=True, help="Folder with source audio (mp3/wav/...)")
    ap.add_argument("--out", required=True, help="Output root folder")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="kicks")
    ap.add_argument("--min_gap", type=float, default=0.10, help="Seconds between transient cuts")
    ap.add_argument("--max_len", type=float, default=1.0, help="Max seconds per slice (None = unlimited)")
    ap.add_argument("--backtrack", type=int, default=1, help="1=enable onset backtrack, 0=disable")
    ap.add_argument("--spec", type=str, default=None, help="JSON string to override thresholds")
    ap.add_argument("--copy_top", type=int, default=64, help="Copy top-N by RMS to filtered folder")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_root = Path(args.out)
    slices_dir = out_root / "slices"
    filtered_dir = out_root / "filtered"
    ensure_dir(slices_dir); ensure_dir(filtered_dir)

    # thresholds
    spec = PRESETS[args.preset].copy()
    if args.spec:
        spec.update(json.loads(args.spec))

    records = []
    files = [p for p in in_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    for src in tqdm(files, desc="Chopping"):
        try:
            y, sr, dur = load_audio(src)
        except Exception as e:
            print(f"[WARN] Could not load {src.name}: {e}")
            continue

        on = detect_onsets(y, sr, backtrack=bool(args.backtrack))
        segs = slice_by_onsets(y, sr, on, min_gap=args.min_gap, max_len_s=(args.max_len if args.max_len>0 else None))

        for i,(s,e) in enumerate(segs):
            seg = y[s:e]
            if len(seg) < int(0.01*sr):  # skip sub-10ms
                continue
            base = src.stem
            out_wav = slices_dir / f"{base}_hit_{i:04d}.wav"
            write_wav(out_wav, seg, sr)

            m = features(seg, sr)
            m.update({
                "file_src": str(src),
                "slice_wav": str(out_wav),
                "index": i,
            })
            records.append(m)

    if not records:
        print("No slices produced. Check input folder or increase max_len / reduce min_gap.")
        return

    df = pd.DataFrame(records)
    csv_path = out_root / "slices_metrics.csv"
    df.to_csv(csv_path, index=False)

    # Filtering by spec
    keep_mask = df.apply(lambda r: should_keep(r, spec), axis=1)
    kept = df[keep_mask].copy()

    # If nothing matched, still exit gracefully with the CSV
    if kept.empty:
        print(f"No slices matched preset '{args.preset}'. CSV saved to: {csv_path}")
        return

    # Copy top-N by RMS (loudness proxy)
    kept = kept.sort_values("rms", ascending=False)
    n = min(args.copy_top, len(kept))
    for _, row in kept.head(n).iterrows():
        src = Path(row["slice_wav"])
        dst = filtered_dir / src.name
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[WARN] Could not copy {src} -> {dst}: {e}")

    kept_csv = out_root / "filtered_metrics.csv"
    kept.head(n).to_csv(kept_csv, index=False)

    print(f"Done.\nAll slices: {slices_dir}\nMetrics CSV: {csv_path}\nFiltered (top {n}): {filtered_dir}\nFiltered CSV: {kept_csv}")

if __name__ == "__main__":
    main()
