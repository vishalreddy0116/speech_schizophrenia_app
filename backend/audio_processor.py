import numpy as np
import librosa

def load_audio(path, sr=16000):
    """Loads an audio file and resamples it to 16000 Hz."""
    y, sr = librosa.load(path, sr=sr)
    return y, sr

def preprocess(y):
    """Applies pre-emphasis and normalizes amplitude to [-1.0, 1.0]."""
    # Pre-emphasis filter
    y_pre = librosa.effects.preemphasis(y)
    # Amplitude normalization
    max_val = np.max(np.abs(y_pre))
    if max_val > 0:
        y_norm = y_pre / (max_val + 1e-6)
    else:
        y_norm = y_pre
    return y_norm

def segment_audio(y, sr=16000, sec=3):
    """Splits audio into fixed-length segments of `sec` seconds, zero-padding the last one."""
    L = sr * sec
    segs = []
    # If audio is empty, return a single silent segment
    if len(y) == 0:
        return [np.zeros(L, dtype=np.float32)]
        
    for i in range(0, len(y), L):
        seg = y[i : i + L]
        if len(seg) < L:
            seg = np.pad(seg, (0, L - len(seg)), 'constant')
        segs.append(seg)
    return segs

def extract_features(y):
    """
    Extracts manual acoustic features for a segment:
      - pause_raw: ratio of near-silent samples (< 0.01)
      - pitch_var: standard deviation of pitch frames calculated via YIN
      - zcr: mean Zero Crossing Rate (speech rate proxy)
      - articulation_raw: inverse of RMS energy (scaled)
    """
    # Safeguard against silent or empty arrays
    if np.max(np.abs(y)) < 0.0001:
        return {
            "pause_raw": 1.0,
            "pitch_var": 0.0,
            "zcr": 0.0,
            "articulation_raw": 1.0
        }

    try:
        # Pitch via YIN (per-frame, Hz)
        pitch_frames = librosa.yin(y, fmin=50, fmax=300)
        pitch_var = float(np.nanstd(pitch_frames))
        if np.isnan(pitch_var):
            pitch_var = 0.0
    except Exception:
        pitch_var = 0.0

    # Energy - RMS
    energy = float(np.mean(librosa.feature.rms(y=y)))

    # ZCR - proxy for speech rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Pause - fraction of near-silent samples
    pause_raw = float(np.mean(np.abs(y) < 0.01))

    # Articulation = 1 - energy (scaled by 10 so it isn't always ~0.99)
    articulation_raw = 1.0 - (energy * 10.0)

    return {
        "pause_raw":        pause_raw,
        "pitch_var":        pitch_var,
        "zcr":              zcr,
        "articulation_raw": articulation_raw,
    }

def compute_ssi(f):
    """Calculates final SSI from the 4-D features dictionary, scaled between 0 and 1."""
    pause        = np.clip(f["pause_raw"],        0.0, 1.0)
    pitch        = float(np.tanh(f["pitch_var"] / 50.0))       # Scaled so 50Hz var = ~0.76
    speech_rate  = float(np.tanh(f["zcr"] * 5.0))              # Scaled so ZCR 0.1 = ~0.46
    articulation = np.clip(f["articulation_raw"], 0.0, 1.0)

    ssi = (pause + pitch + speech_rate + articulation) / 4.0
    return float(np.clip(ssi, 0.0, 1.0))

def articulatory_matrix(f):
    """Converts the extracted features into a 4D tensor input for VQ-VAE."""
    return np.array([
        np.clip(f["pause_raw"],        0.0, 1.0),
        float(np.tanh(f["pitch_var"] / 50.0)),
        float(np.tanh(f["zcr"] * 5.0)),
        np.clip(f["articulation_raw"], 0.0, 1.0),
    ], dtype=np.float32)
