"""
Domain Profile Selector for ReVox Voice Privacy Deployment Scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np


@dataclass(frozen=True)
class DomainProfileSpec:
    profile_id: str
    display_name: str
    short_description: str
    preserve_gender: bool | None
    default_knn_topk: int
    allow_disable_rvox: bool
    default_embed_rvox: bool


DOMAIN_PROFILES: dict[str, DomainProfileSpec] = {
    "banking_call_center": DomainProfileSpec(
        profile_id="banking_call_center",
        display_name="🏦 Tổng đài Ngân hàng (Giữ giới tính & ngữ điệu)",
        short_description="Chọn giọng xa nhất cùng giới tính, nhúng khóa pháp lý RVOX.",
        preserve_gender=True,
        default_knn_topk=4,
        allow_disable_rvox=False,
        default_embed_rvox=True,
    ),
    "medical_telehealth": DomainProfileSpec(
        profile_id="medical_telehealth",
        display_name="🩺 Y tế & Tư vấn Tâm lý (Giọng trầm ấm, mượt)",
        short_description="Chọn giọng trầm ấm, mượt phổ cho hồ sơ bệnh án dài.",
        preserve_gender=None,
        default_knn_topk=6,
        allow_disable_rvox=False,
        default_embed_rvox=True,
    ),
    "whistleblower_max": DomainProfileSpec(
        profile_id="whistleblower_max",
        display_name="🕵️ Bảo vệ Nhân chứng (Đổi chéo giới tính tối đa)",
        short_description="Đổi chéo giới tính Nam <-> Nữ, hỗ trợ khóa tư pháp hoặc ẩn danh 1 chiều.",
        preserve_gender=False,
        default_knn_topk=4,
        allow_disable_rvox=True,
        default_embed_rvox=True,
    ),
}


def estimate_gender_f0(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    threshold_hz: float = 165.0,
) -> tuple[str, float]:
    wav = np.asarray(waveform, dtype=np.float32).squeeze()
    if wav.size < sample_rate // 4:
        return "female", 180.0

    f0, _, _ = librosa.pyin(
        wav,
        fmin=65.0,
        fmax=360.0,
        sr=sample_rate,
        frame_length=1024,
        hop_length=256,
    )
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if voiced_f0.size < 3:
        f0_yin = librosa.yin(wav, fmin=65.0, fmax=360.0, sr=sample_rate, frame_length=1024, hop_length=256)
        valid_yin = f0_yin[(f0_yin > 70.0) & (f0_yin < 340.0)]
        median_f0 = float(np.median(valid_yin)) if valid_yin.size > 0 else 165.0
    else:
        median_f0 = float(np.median(voiced_f0))

    gender = "male" if median_f0 < threshold_hz else "female"
    return gender, round(median_f0, 2)


def select_target_speaker(
    input_ecapa_embedding: np.ndarray,
    input_gender: str,
    pool_entries: list[dict[str, Any]],
    domain_profile: str = "banking_call_center",
    exclude_speaker_id: str | None = None,
) -> dict[str, Any]:
    if domain_profile not in DOMAIN_PROFILES:
        raise ValueError(f"Kịch bản '{domain_profile}' không hợp lệ.")
    if not pool_entries:
        raise RuntimeError("Bể giọng đích (target_pool) đang trống!")

    spec = DOMAIN_PROFILES[domain_profile]
    norm_in = input_ecapa_embedding.astype(np.float32).squeeze()
    norm_in = norm_in / (np.linalg.norm(norm_in) + 1e-8)
    in_gender = input_gender.strip().lower()

    candidates = [
        entry
        for entry in pool_entries
        if exclude_speaker_id is None or entry["speaker_id"] != exclude_speaker_id
    ] or list(pool_entries)

    scored_candidates: list[tuple[float, float, dict[str, Any]]] = []
    for entry in candidates:
        emb = np.asarray(entry["ecapa_embedding"], dtype=np.float32).squeeze()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        cos_sim = float(np.dot(norm_in, emb))
        scored_candidates.append((cos_sim, 1.0 - cos_sim, entry))

    if domain_profile == "banking_call_center":
        same_gender = [item for item in scored_candidates if item[2].get("gender", "").lower() == in_gender]
        active = same_gender if same_gender else scored_candidates
        best_cos, _, chosen = min(active, key=lambda x: x[0])
        reason = f"Cùng giới tính ({chosen['gender']}), ECAPA xa nhất (Cos={best_cos:.3f})"

    elif domain_profile == "whistleblower_max":
        target_gender = "female" if in_gender == "male" else "male"
        opp_gender = [item for item in scored_candidates if item[2].get("gender", "").lower() == target_gender]
        active = opp_gender if opp_gender else scored_candidates
        best_cos, _, chosen = min(active, key=lambda x: x[0])
        reason = f"Đổi chéo giới tính ({in_gender} -> {chosen['gender']}), ECAPA xa nhất (Cos={best_cos:.3f})"

    else:
        sorted_by_sim = sorted(scored_candidates, key=lambda x: x[0])
        cutoff = max(1, int(np.ceil(len(sorted_by_sim) * 0.6)))
        safe_pool = sorted_by_sim[:cutoff]

        def _medical_utility(item: tuple[float, float, dict[str, Any]]) -> float:
            _, privacy_dist, ent = item
            smooth = float(ent.get("spectral_smoothness", 0.75))
            warmth = float(ent.get("warmth_score", 0.75))
            gender_bonus = 0.05 if ent.get("gender", "").lower() == in_gender else 0.0
            return 0.45 * smooth + 0.35 * warmth + 0.20 * privacy_dist + gender_bonus

        best_cos, _, chosen = max(safe_pool, key=_medical_utility)
        reason = f"Giọng trầm ấm & mượt ({chosen['gender']}, F0={chosen.get('median_f0_hz', 0):.0f}Hz)"

    return {
        "target_entry": chosen,
        "target_speaker_id": chosen["speaker_id"],
        "target_gender": chosen["gender"],
        "pool_cosine_similarity": round(best_cos, 4),
        "knn_topk": spec.default_knn_topk,
        "selection_reason": reason,
        "profile_spec": spec,
    }
