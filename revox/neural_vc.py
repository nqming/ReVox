"""
Core Neural Voice Conversion Engine V2.0 for ReVox (`WavLM-Large` Layer 6 + Triphone `kNN-VC` + `HiFi-GAN`).

Enhancements for Crystal-Clear Vietnamese Word & 6-Tone Intelligibility:
1. Dense Multi-Corpus Target Matching Clouds (up to 7,500 voiced frames / ~2.5 min per target speaker)
2. Triphone Temporal-Context Cosine Matching (`[t-1, t, t+1]` diagonal coarticulation alignment)
   eliminating syllable/diphthong fragmentation and consonant slurring
3. Adaptive AdaIN Phonetic Anchor Blending preserving rare words, names, and plosive bursts while
   maintaining complete speaker identity anonymization on ECAPA-TDNN.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from revox import MODELS_CACHE_DIR
from revox.speaker_biometrics import TARGET_SR, get_optimal_device, load_mono_16k

WAVLM_FEATURE_LAYER = 6
WAVLM_DIM = 1024
HOP_LENGTH = 320  # 20ms @ 16kHz -> 50 frames/sec


@contextlib.contextmanager
def _allow_legacy_checkpoint_unpickle():
    orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        yield
    finally:
        torch.load = orig_load


def trim_nonspeech_edges(
    waveform: np.ndarray,
    top_db: float = 28.0,
) -> np.ndarray:
    wav = np.asarray(waveform, dtype=np.float32).squeeze()
    if wav.size < TARGET_SR // 5:
        return wav
    intervals = librosa.effects.split(wav, top_db=top_db, frame_length=1024, hop_length=256)
    if len(intervals) == 0:
        return wav
    voiced_segments = [wav[start:end] for start, end in intervals if (end - start) >= 320]
    if not voiced_segments:
        return wav
    return np.concatenate(voiced_segments, axis=0).astype(np.float32)


class NeuralVCEngine:
    """High-intelligibility WavLM-Large (Layer 6) + Triphone kNN-VC + HiFi-GAN Engine."""

    def __init__(self, device: str | None = None, num_cpu_threads: int | None = None) -> None:
        self.device = get_optimal_device(device)
        if self.device == "cpu":
            optimal_threads = num_cpu_threads or min(8, max(2, (os.cpu_count() or 4) // 2))
            torch.set_num_threads(optimal_threads)

        hub_dir = MODELS_CACHE_DIR / "torch_hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))

        local_repo = self._find_cached_knnvc_repo(hub_dir)
        with _allow_legacy_checkpoint_unpickle():
            if local_repo is not None:
                self.knn_vc = torch.hub.load(
                    str(local_repo),
                    "knn_vc",
                    source="local",
                    prematched=True,
                    pretrained=True,
                    progress=True,
                    device=self.device,
                )
            else:
                self.knn_vc = torch.hub.load(
                    "bshall/knn-vc",
                    "knn_vc",
                    prematched=True,
                    trust_repo=True,
                    pretrained=True,
                    progress=True,
                    device=self.device,
                )

        self.knn_vc.eval()
        self.wavlm = self.knn_vc.wavlm
        self.hifigan = self.knn_vc.hifigan

    @staticmethod
    def _find_cached_knnvc_repo(hub_dir: Path) -> Path | None:
        checkpoints_dir = hub_dir / "checkpoints"
        wavlm_ckpt = checkpoints_dir / "WavLM-Large.pt"
        hifigan_ckpt = checkpoints_dir / "prematch_g_02500000.pt"
        if not (wavlm_ckpt.exists() and hifigan_ckpt.exists()):
            return None
        for candidate in hub_dir.glob("bshall_knn-vc_*"):
            if (candidate / "hubconf.py").exists():
                return candidate
        return None

    @torch.inference_mode()
    def extract_wavlm_features(
        self,
        audio_input: str | Path | np.ndarray | torch.Tensor,
        sample_rate: int = TARGET_SR,
        trim_silence: bool = False,
    ) -> torch.Tensor:
        wav = load_mono_16k(audio_input, sr=sample_rate)
        if trim_silence:
            wav = trim_nonspeech_edges(wav)
        if wav.size < HOP_LENGTH * 2:
            wav = np.pad(wav, (0, HOP_LENGTH * 2 - wav.size))

        wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        feats = self.knn_vc.get_features(wav_tensor, vad_trigger_level=0)
        if feats.ndim == 3:
            feats = feats.squeeze(0)
        return feats.float()

    @torch.inference_mode()
    def extract_matching_pool(
        self,
        utterance_paths: Sequence[str | Path],
        max_frames: int = 8500,
    ) -> torch.Tensor:
        """
        Extract clean per-utterance WavLM-Large Layer-6 frames (untrimmed so unvoiced consonants
        k/t/p/s/kh/th, stop closures, and utterance-level LayerNorm stay 100% intact).
        """
        chunks: list[torch.Tensor] = []
        total_frames = 0
        for p in utterance_paths:
            feats = self.extract_wavlm_features(p, trim_silence=False)
            chunks.append(feats.cpu())
            total_frames += int(feats.shape[0])
            if total_frames >= max_frames:
                break

        if not chunks:
            raise ValueError("Không trích xuất được đặc trưng WavLM từ danh sách file giọng đích.")
        pool = torch.cat(chunks, dim=0)
        if pool.shape[0] > max_frames:
            pool = pool[:max_frames]
        return pool.to(self.device, dtype=torch.float32)

    @staticmethod
    @torch.inference_mode()
    def knn_convert_features(
        query_features: torch.Tensor,
        matching_pool: torch.Tensor,
        topk: int = 4,
        triphone_context: bool = True,
        phonetic_anchor_min: float = 0.25,
        phonetic_anchor_max: float = 0.40,
        std_floor: float = 0.20,
    ) -> torch.Tensor:
        """
        High-intelligibility Triphone Context kNN-VC with Regularized AdaIN Phonetic Preservation:
        1. Hybrid raw (65%) + speaker-demeaned (35%) cosine similarity removes vocal-tract bias.
        2. Gentle 3-frame diagonal triphone smoothing `[0.15, 0.70, 0.15]` preserves coarticulation & tones.
        3. Variance-floored AdaIN phonetic guidance keeps consonants, diphthongs & rare words crisp
           while preserving strong ECAPA-TDNN speaker anonymization.
        """
        q = query_features.squeeze(0).float()
        m = matching_pool.squeeze(0).to(device=q.device, dtype=torch.float32)

        q_norm = F.normalize(q, p=2, dim=-1)
        m_norm = F.normalize(m, p=2, dim=-1)
        sim_raw = torch.matmul(q_norm, m_norm.transpose(0, 1))

        mu_q = q.mean(dim=0, keepdim=True)
        mu_m = m.mean(dim=0, keepdim=True)
        q_dem = F.normalize(q - mu_q, p=2, dim=-1)
        m_dem = F.normalize(m - mu_m, p=2, dim=-1)
        sim_dem = torch.matmul(q_dem, m_dem.transpose(0, 1))

        sim = 0.65 * sim_raw + 0.35 * sim_dem

        if triphone_context and sim.shape[0] >= 3 and sim.shape[1] >= 3:
            sim_ctx = sim.clone()
            sim_ctx[1:-1, 1:-1] = (
                0.15 * sim[:-2, :-2]
                + 0.70 * sim[1:-1, 1:-1]
                + 0.15 * sim[2:, 2:]
            )
            sim = sim_ctx

        eff_k = min(int(topk), int(m.shape[0]))
        top_sims, top_indices = torch.topk(sim, k=eff_k, dim=-1, largest=True, sorted=True)

        knn_matched = m[top_indices].mean(dim=1)

        if phonetic_anchor_max > 0.0:
            std_q = torch.clamp(q.std(dim=0, keepdim=True), min=std_floor)
            std_m = torch.clamp(m.std(dim=0, keepdim=True), min=std_floor)
            q_adain = ((q - mu_q) / std_q) * std_m + mu_m

            best_sim = top_sims[:, 0:1].clamp(0.45, 0.90)
            lack_of_match = (0.90 - best_sim) / 0.45
            beta = phonetic_anchor_min + (phonetic_anchor_max - phonetic_anchor_min) * lack_of_match
            matched = (1.0 - beta) * knn_matched + beta * q_adain
        else:
            matched = knn_matched

        return matched

    @torch.inference_mode()
    def vocode_features(self, wavlm_features: torch.Tensor) -> np.ndarray:
        feats = wavlm_features.to(device=self.device, dtype=torch.float32)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)

        wav_out = self.hifigan(feats).squeeze().detach().cpu().float().numpy()
        peak = float(np.max(np.abs(wav_out))) if wav_out.size > 0 else 0.0
        if peak > 0.99:
            wav_out = wav_out * (0.98 / peak)
        return wav_out.astype(np.float32)
