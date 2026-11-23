"""
Speaker Biometrics & Verification Module for ReVox (`speechbrain/spkrec-ecapa-voxceleb`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import roc_curve

from revox import MODELS_CACHE_DIR, _apply_runtime_compat_patches

_apply_runtime_compat_patches()

from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
from speechbrain.utils.fetching import LocalStrategy  # noqa: E402

TARGET_SR = 16000
ECAPA_REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"


def get_optimal_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_mono_16k(audio_input: str | Path | np.ndarray | torch.Tensor, sr: int = TARGET_SR) -> np.ndarray:
    if isinstance(audio_input, (str, Path)):
        wav, orig_sr = sf.read(str(audio_input), dtype="float32", always_2d=False)
    elif isinstance(audio_input, torch.Tensor):
        wav = audio_input.detach().cpu().float().numpy()
        orig_sr = sr
    else:
        wav = np.asarray(audio_input, dtype=np.float32)
        orig_sr = sr

    if wav.ndim == 2:
        if wav.shape[0] <= 4 and wav.shape[1] > wav.shape[0]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=1)
    wav = wav.squeeze().astype(np.float32)

    if orig_sr != TARGET_SR and wav.size > 0:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        resampled = torchaudio.functional.resample(wav_tensor, orig_freq=orig_sr, new_freq=TARGET_SR)
        wav = resampled.squeeze(0).numpy().astype(np.float32)

    return wav


class SpeakerBiometricsEngine:
    def __init__(self, device: str | None = None, cache_dir: str | Path | None = None) -> None:
        self.device = get_optimal_device(device)
        save_dir = Path(cache_dir) if cache_dir else (MODELS_CACHE_DIR / "ecapa_tdnn")
        save_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = EncoderClassifier.from_hparams(
            source=ECAPA_REPO_ID,
            savedir=str(save_dir),
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )
        self.classifier.eval()

    @torch.inference_mode()
    def extract_embedding(
        self,
        audio_input: str | Path | np.ndarray | torch.Tensor,
        sample_rate: int = TARGET_SR,
    ) -> np.ndarray:
        wav = load_mono_16k(audio_input, sr=sample_rate)
        if wav.size < 400:
            raise ValueError("Tín hiệu âm thanh quá ngắn (< 25ms) để trích xuất ECAPA-TDNN.")

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        wav_lens = torch.ones(1, device=self.device)
        emb = self.classifier.encode_batch(wav_tensor, wav_lens)
        emb_np = emb.squeeze().detach().cpu().float().numpy()
        norm = float(np.linalg.norm(emb_np))
        if norm > 1e-8:
            emb_np = emb_np / norm
        return emb_np.astype(np.float32)


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    a = np.asarray(emb_a, dtype=np.float32).squeeze()
    b = np.asarray(emb_b, dtype=np.float32).squeeze()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def biometric_match_percent(cos_sim: float) -> float:
    return round(float(np.clip(cos_sim, 0.0, 1.0) * 100.0), 2)


def compute_eer(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
) -> dict[str, Any]:
    pos = np.asarray(genuine_scores, dtype=np.float64).ravel()
    neg = np.asarray(impostor_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Cần ít nhất 1 cặp genuine và 1 cặp impostor để tính EER.")

    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    diff = fpr - fnr
    idx = int(np.nanargmin(np.abs(diff)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    for i in range(len(diff) - 1):
        if diff[i] == 0.0:
            eer = float(fpr[i])
            eer_threshold = float(thresholds[i])
            break
        if (diff[i] < 0.0 and diff[i + 1] > 0.0) or (diff[i] > 0.0 and diff[i + 1] < 0.0):
            alpha = -diff[i] / (diff[i + 1] - diff[i])
            eer = float(fpr[i] + alpha * (fpr[i + 1] - fpr[i]))
            if np.isfinite(thresholds[i]) and np.isfinite(thresholds[i + 1]):
                eer_threshold = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
            break

    return {
        "eer": eer,
        "eer_percent": round(eer * 100.0, 2),
        "threshold": round(eer_threshold, 4),
        "fpr": fpr,
        "fnr": fnr,
        "thresholds": thresholds,
    }
