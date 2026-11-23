"""
Reversible Acoustic Codebook Engine for ReVox (K=128 MiniBatchKMeans + float16 + LZMA).
"""

from __future__ import annotations

import io
import json
import lzma
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

CODEBOOK_MAGIC = b"RVCB"
CODEBOOK_VERSION = 1
DEFAULT_NUM_CENTROIDS = 128


@dataclass
class TimbreCodebook:
    """Personal acoustic timbre codebook extracted from source speaker's WavLM features."""

    centroids: np.ndarray
    utterance_mean: np.ndarray
    utterance_std: np.ndarray
    source_frames: int
    feature_dim: int
    metadata: dict[str, Any]

    @property
    def num_centroids(self) -> int:
        return int(self.centroids.shape[0])


def build_timbre_codebook(
    wavlm_features: torch.Tensor | np.ndarray,
    n_clusters: int = DEFAULT_NUM_CENTROIDS,
    metadata: dict[str, Any] | None = None,
    random_state: int = 42,
) -> TimbreCodebook:
    if isinstance(wavlm_features, torch.Tensor):
        feats = wavlm_features.detach().cpu().float().numpy()
    else:
        feats = np.asarray(wavlm_features, dtype=np.float32)

    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats[0]
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D WavLM features (T, D), got shape {feats.shape}")

    num_frames, feat_dim = feats.shape
    if num_frames == 0:
        raise ValueError("Chuỗi đặc trưng WavLM rỗng, không thể tạo bảng mã khôi phục.")

    utt_mean = feats.mean(axis=0).astype(np.float32)
    utt_std = (feats.std(axis=0) + 1e-6).astype(np.float32)

    effective_k = min(int(n_clusters), num_frames)
    if effective_k >= num_frames:
        centroids = feats.copy().astype(np.float32)
    else:
        batch_size = max(256, effective_k * 2)
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=batch_size,
            n_init=3,
            max_iter=150,
            random_state=random_state,
            reassignment_ratio=0.02,
        )
        kmeans.fit(feats)
        centroids = kmeans.cluster_centers_.astype(np.float32)

    return TimbreCodebook(
        centroids=centroids,
        utterance_mean=utt_mean,
        utterance_std=utt_std,
        source_frames=int(num_frames),
        feature_dim=int(feat_dim),
        metadata=dict(metadata or {}),
    )


def serialize_codebook(codebook: TimbreCodebook, lzma_preset: int = 6) -> bytes:
    k, d = codebook.centroids.shape
    meta_bytes = json.dumps(codebook.metadata, ensure_ascii=False).encode("utf-8")

    buf = io.BytesIO()
    buf.write(CODEBOOK_MAGIC)
    buf.write(
        struct.pack(
            "<BHHII",
            CODEBOOK_VERSION,
            int(k),
            int(d),
            int(codebook.source_frames),
            len(meta_bytes),
        )
    )
    buf.write(meta_bytes)
    buf.write(codebook.utterance_mean.astype(np.float16).tobytes())
    buf.write(codebook.utterance_std.astype(np.float16).tobytes())
    buf.write(codebook.centroids.astype(np.float16).tobytes())

    return lzma.compress(buf.getvalue(), format=lzma.FORMAT_XZ, preset=lzma_preset)


def deserialize_codebook(compressed_bytes: bytes) -> TimbreCodebook:
    try:
        raw_bytes = lzma.decompress(compressed_bytes)
    except lzma.LZMAError as exc:
        raise ValueError("Không thể giải nén LZMA bảng mã âm sắc RVOX.") from exc

    header_fmt = "<BHHII"
    header_size = 4 + struct.calcsize(header_fmt)
    if len(raw_bytes) < header_size or raw_bytes[:4] != CODEBOOK_MAGIC:
        raise ValueError("Header bảng mã âm sắc RVCB không hợp lệ.")

    version, k, d, source_frames, meta_len = struct.unpack(header_fmt, raw_bytes[4:header_size])
    if version != CODEBOOK_VERSION:
        raise ValueError(f"Phiên bản bảng mã âm sắc không được hỗ trợ: v{version}")

    pos = header_size
    meta_json = raw_bytes[pos : pos + meta_len].decode("utf-8")
    metadata = json.loads(meta_json) if meta_json else {}
    pos += meta_len

    vec_bytes = d * 2
    utt_mean = np.frombuffer(raw_bytes[pos : pos + vec_bytes], dtype=np.float16).astype(np.float32).copy()
    pos += vec_bytes
    utt_std = np.frombuffer(raw_bytes[pos : pos + vec_bytes], dtype=np.float16).astype(np.float32).copy()
    pos += vec_bytes

    centroids_bytes = k * d * 2
    centroids = (
        np.frombuffer(raw_bytes[pos : pos + centroids_bytes], dtype=np.float16)
        .reshape(k, d)
        .astype(np.float32)
        .copy()
    )

    return TimbreCodebook(
        centroids=centroids,
        utterance_mean=utt_mean,
        utterance_std=utt_std,
        source_frames=int(source_frames),
        feature_dim=int(d),
        metadata=metadata,
    )


@torch.inference_mode()
def reverse_knn_reconstruct(
    anon_features: torch.Tensor,
    codebook: TimbreCodebook,
    topk: int = 4,
    temperature: float = 0.025,
    demean_weight: float = 0.65,
) -> torch.Tensor:
    """
    Map WavLM features Z_anon (T', D) back onto source speaker's K-centroid codebook C_src (K, D)
    with speaker-demeaned hybrid Soft-kNN + AdaIN fine phonetic detail preservation.
    """
    device = anon_features.device
    z_anon = anon_features.squeeze(0).float() if anon_features.ndim == 3 else anon_features.float()
    c_src = torch.from_numpy(codebook.centroids).to(device=device, dtype=torch.float32)
    mu_src = torch.from_numpy(codebook.utterance_mean).to(device=device, dtype=torch.float32)
    std_src = torch.from_numpy(codebook.utterance_std).to(device=device, dtype=torch.float32)

    z_norm = F.normalize(z_anon, p=2, dim=-1)
    c_norm = F.normalize(c_src, p=2, dim=-1)
    sim_raw = torch.matmul(z_norm, c_norm.transpose(0, 1))

    mu_anon = z_anon.mean(dim=0, keepdim=True)
    std_anon = z_anon.std(dim=0, keepdim=True) + 1e-6
    z_centered = F.normalize(z_anon - mu_anon, p=2, dim=-1)
    c_centered = F.normalize(c_src - mu_src.unsqueeze(0), p=2, dim=-1)
    sim_phonetic = torch.matmul(z_centered, c_centered.transpose(0, 1))

    sim_hybrid = demean_weight * sim_phonetic + (1.0 - demean_weight) * sim_raw

    eff_k = min(int(topk), c_src.shape[0])
    top_sims, top_idx = torch.topk(sim_hybrid, k=eff_k, dim=-1, largest=True, sorted=True)

    if eff_k == 1 or temperature <= 0.0:
        knn_rec = c_src[top_idx[:, 0]]
    else:
        weights = torch.softmax(top_sims / temperature, dim=-1)
        knn_rec = torch.sum(weights.unsqueeze(-1) * c_src[top_idx], dim=1)

    # Blend 15% AdaIN-aligned trajectory so consonant transitions stay crisp alongside 85% codebook timbre
    z_adain = ((z_anon - mu_anon) / std_anon) * std_src.unsqueeze(0) + mu_src.unsqueeze(0)
    return 0.85 * knn_rec + 0.15 * z_adain
