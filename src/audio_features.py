from __future__ import annotations
import numpy as np
import torch
import librosa


class AudioFeatureExtractor:
    """Extract segment-level log-mel, chroma and MFCC summaries.

    Unlike the original notebook, dimensions are not standardized against each
    other inside every node. A train-split scaler should be fitted later.
    """
    def __init__(self, sr=22050, n_mels=64, n_mfcc=20, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.feature_dim = n_mels + 12 + n_mfcc + 4

    def _one(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            hop_length=self.hop_length, power=2.0
        )
        logmel = librosa.power_to_db(mel + 1e-10, ref=np.max).mean(axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, hop_length=self.hop_length).mean(axis=1)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                                    hop_length=self.hop_length).mean(axis=1)
        rms = float(librosa.feature.rms(y=y).mean())
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=self.sr).mean())
        rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=self.sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        return np.concatenate([logmel, chroma, mfcc, [rms, centroid, rolloff, zcr]]).astype(np.float32)

    def extract_segments(self, y: np.ndarray, window_seconds=2.0, hop_seconds=1.0) -> torch.Tensor:
        win = int(window_seconds * self.sr)
        hop = int(hop_seconds * self.sr)
        if y.size < win:
            y = np.pad(y, (0, win - y.size))
        starts = list(range(0, max(1, y.size - win + 1), hop))
        if not starts:
            starts = [0]
        feats = [self._one(y[s:s+win]) for s in starts]
        return torch.tensor(np.stack(feats), dtype=torch.float32)

    def load_clip(self, path: str, duration: float | None = None, offset: float = 0.0):
        y, _ = librosa.load(path, sr=self.sr, mono=True, duration=duration, offset=offset)
        return y
