"""
Scientific Evaluation & Acoustic Visualization Module for ReVox.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from faster_whisper import WhisperModel

from revox import MODELS_CACHE_DIR
from revox.speaker_biometrics import TARGET_SR, compute_eer, cosine_similarity, load_mono_16k

_VI_TONE_CANONICAL_MAP = {
    "oà": "òa",
    "oá": "óa",
    "oả": "ỏa",
    "oã": "õa",
    "oạ": "ọa",
    "oè": "òe",
    "oé": "óe",
    "oẻ": "ỏe",
    "oẽ": "õe",
    "oẹ": "ọe",
    "uỳ": "ùy",
    "uý": "úy",
    "uỷ": "ủy",
    "uỹ": "ũy",
    "uỵ": "ụy",
}


def normalize_text_for_wer(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"_+", " ", s)
    for src_pat, dst_pat in _VI_TONE_CANONICAL_MAP.items():
        s = s.replace(src_pat, dst_pat)
    return re.sub(r"\s+", " ", s).strip()


def _levenshtein_distance(seq_a: list[Any], seq_b: list[Any]) -> int:
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    if n < m:
        seq_a, seq_b = seq_b, seq_a
        n, m = m, n

    prev_row = list(range(m + 1))
    for i in range(1, n + 1):
        curr_row = [i] + [0] * m
        ca = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ca == seq_b[j - 1] else 1
            curr_row[j] = min(curr_row[j - 1] + 1, prev_row[j] + 1, prev_row[j - 1] + cost)
        prev_row = curr_row
    return prev_row[m]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text_for_wer(reference).split()
    hyp_words = normalize_text_for_wer(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return float(_levenshtein_distance(ref_words, hyp_words)) / float(len(ref_words))


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text_for_wer(reference).replace(" ", ""))
    hyp_chars = list(normalize_text_for_wer(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return float(_levenshtein_distance(ref_chars, hyp_chars)) / float(len(ref_chars))


def compute_realtime_metrics(processing_sec: float, audio_duration_sec: float) -> dict[str, float]:
    dur = max(audio_duration_sec, 1e-4)
    proc = max(processing_sec, 1e-5)
    return {
        "processing_sec": round(proc, 4),
        "audio_duration_sec": round(dur, 4),
        "rtf": round(proc / dur, 4),
        "xrt": round(dur / proc, 2),
    }


class FasterWhisperEvaluator:
    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        cache_root = MODELS_CACHE_DIR / "faster_whisper"
        local_model_dir = cache_root / model_size
        cache_root.mkdir(parents=True, exist_ok=True)
        cpu_threads = min(8, max(2, (os.cpu_count() or 4) // 2))

        if not (local_model_dir / "model.bin").exists():
            import huggingface_hub

            huggingface_hub.snapshot_download(
                repo_id=f"Systran/faster-whisper-{model_size}",
                local_dir=str(local_model_dir),
            )

        self.model_size = model_size
        self.device = device
        self.model = WhisperModel(
            str(local_model_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(
        self,
        audio_input: str | Path | np.ndarray,
        language: str | None = "vi",
        beam_size: int = 4,
    ) -> dict[str, Any]:
        wav = load_mono_16k(audio_input, sr=TARGET_SR)
        use_vad = (float(wav.size) / float(TARGET_SR)) > 15.0
        vad_kwargs = {"vad_parameters": {"min_silence_duration_ms": 350}} if use_vad else {}
        segments, info = self.model.transcribe(
            wav,
            language=language,
            beam_size=beam_size,
            vad_filter=use_vad,
            condition_on_previous_text=False,
            **vad_kwargs,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {
            "text": unicodedata.normalize("NFC", text),
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
        }


def _compute_log_mel(wav: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=1024, hop_length=256, n_mels=80, fmin=40, fmax=sr // 2
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_mel_pair(
    orig_audio: str | Path | np.ndarray,
    anon_audio: str | Path | np.ndarray,
    sr: int = TARGET_SR,
    cos_sim: float | None = None,
    profile_name: str = "",
) -> plt.Figure:
    wav_orig = load_mono_16k(orig_audio, sr=sr)
    wav_anon = load_mono_16k(anon_audio, sr=sr)
    mel_orig = _compute_log_mel(wav_orig, sr=sr)
    mel_anon = _compute_log_mel(wav_anon, sr=sr)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("#0f172a")

    panels = [
        (axes[0], mel_orig, "Giọng Gốc"),
        (
            axes[1],
            mel_anon,
            "Giọng Đã Ẩn Danh" + (f" (Cos = {cos_sim:.3f})" if cos_sim is not None else ""),
        ),
    ]

    im = None
    for ax, mel_db, title in panels:
        ax.set_facecolor("#1e293b")
        im = librosa.display.specshow(
            mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", fmax=8000, cmap="magma", ax=ax
        )
        ax.set_title(title, color="#f8fafc", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("Thời gian (s)", color="#cbd5e1", fontsize=9)
        ax.set_ylabel("Hz", color="#cbd5e1", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format="%+2.0f dB", shrink=0.88)
        cbar.ax.yaxis.set_tick_params(color="#cbd5e1", labelcolor="#cbd5e1", labelsize=8)

    return fig


def plot_mel_trio(
    anon_audio: str | Path | np.ndarray,
    recon_audio: str | Path | np.ndarray,
    orig_audio: str | Path | np.ndarray | None = None,
    sr: int = TARGET_SR,
    anon_cos: float | None = None,
    recon_cos: float | None = None,
) -> plt.Figure:
    wav_anon = load_mono_16k(anon_audio, sr=sr)
    wav_recon = load_mono_16k(recon_audio, sr=sr)
    mel_anon = _compute_log_mel(wav_anon, sr=sr)
    mel_recon = _compute_log_mel(wav_recon, sr=sr)

    if orig_audio is not None:
        wav_orig = load_mono_16k(orig_audio, sr=sr)
        mel_orig = _compute_log_mel(wav_orig, sr=sr)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
        panels = [
            (axes[0], mel_orig, "Giọng Gốc"),
            (
                axes[1],
                mel_anon,
                "Giọng Ẩn Danh" + (f" (Cos={anon_cos:.3f})" if anon_cos is not None else ""),
            ),
            (
                axes[2],
                mel_recon,
                "Giọng Khôi Phục" + (f" (Cos={recon_cos:.3f})" if recon_cos is not None else ""),
            ),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
        panels = [
            (axes[0], mel_anon, "Giọng Ẩn Danh"),
            (axes[1], mel_recon, "Giọng Khôi Phục"),
        ]

    fig.patch.set_facecolor("#0f172a")
    im = None
    for ax, mel_db, title in panels:
        ax.set_facecolor("#1e293b")
        im = librosa.display.specshow(
            mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", fmax=8000, cmap="magma", ax=ax
        )
        ax.set_title(title, color="#f8fafc", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Thời gian (s)", color="#cbd5e1", fontsize=9)
        ax.set_ylabel("Hz", color="#cbd5e1", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), format="%+2.0f dB", shrink=0.88)
        cbar.ax.yaxis.set_tick_params(color="#cbd5e1", labelcolor="#cbd5e1", labelsize=8)

    return fig
