"""HRV, SpO2, respiratory feature extraction utilities."""
import numpy as np

def extract_hrv_features(rr_ms: np.ndarray) -> np.ndarray:
    if len(rr_ms) < 2: return np.zeros(4, dtype=np.float32)
    diff = np.diff(rr_ms)
    return np.array([np.mean(rr_ms), np.sqrt(np.mean(diff**2)), np.mean(np.abs(diff)>50)*100, np.std(rr_ms)], dtype=np.float32)

def extract_spo2_features(sig: np.ndarray) -> np.ndarray:
    return np.array([(float(np.clip(np.mean(sig),70,100))-70)/30.0], dtype=np.float32)

def extract_resp_features(sig: np.ndarray, fs: float = 25.0) -> np.ndarray:
    crossings = np.where(np.diff(np.sign(sig - np.mean(sig))))[0]
    rate = float(len(crossings)) / (2.0 * len(sig) / fs)
    return np.array([rate], dtype=np.float32)
