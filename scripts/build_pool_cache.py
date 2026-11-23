"""
Precompute Dense WavLM-Large (Layer 6) + ECAPA-TDNN Cache for `data/target_pool/` (`scripts/build_pool_cache.py`).

Uses chunked batch extraction (~20s concatenated voiced segments per forward pass) for fast CPU/GPU
extraction of ~4,800-5,500 voiced frames per target speaker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revox import MODELS_CACHE_DIR, TARGET_POOL_DIR  # noqa: E402
from revox.domain_selector import estimate_gender_f0  # noqa: E402
from revox.neural_vc import NeuralVCEngine, trim_nonspeech_edges  # noqa: E402
from revox.speaker_biometrics import (  # noqa: E402
    TARGET_SR,
    SpeakerBiometricsEngine,
    load_mono_16k,
)


def _compute_acoustic_warmth_and_smoothness(
    wavs: list[np.ndarray],
    gender: str,
    median_f0: float,
) -> tuple[float, float]:
    concat = np.concatenate(wavs[:5], axis=0) if wavs else np.zeros(TARGET_SR, dtype=np.float32)
    if concat.size < TARGET_SR // 2:
        return 0.75, 0.75

    mel = librosa.feature.melspectrogram(y=concat, sr=TARGET_SR, n_fft=1024, hop_length=256, n_mels=80)
    log_mel = librosa.power_to_db(mel + 1e-9, ref=np.max)
    mel_diff = np.diff(log_mel, axis=1)
    mean_flux = float(np.mean(np.linalg.norm(mel_diff, axis=0)))
    raw_smoothness = float(1.0 / (1.0 + 0.04 * mean_flux))

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=concat, n_fft=1024, hop_length=256)))
    harmonic_purity = float(np.clip(1.0 - flatness * 4.0, 0.2, 1.0))

    ideal_warm_f0 = 122.0 if gender == "male" else 192.0
    pitch_warmth = float(np.exp(-((median_f0 - ideal_warm_f0) ** 2) / (2.0 * (35.0**2))))
    raw_warmth = 0.55 * harmonic_purity + 0.45 * pitch_warmth
    return raw_smoothness, raw_warmth


def _extract_chunked_wavlm_pool(
    vc: NeuralVCEngine,
    wavs: list[np.ndarray],
    max_frames: int = 5200,
) -> torch.Tensor:
    """
    Concatenate trimmed voiced utterances into ~18-second macro-chunks (288,000 samples) to minimize
    per-file convolution/padding overhead while preserving natural within-utterance triphone transitions.
    """
    trimmed = [trim_nonspeech_edges(w) for w in wavs]
    trimmed = [w for w in trimmed if w.size >= 1600]
    if not trimmed:
        trimmed = wavs[:5]

    chunk_samples_limit = TARGET_SR * 18
    macro_chunks: list[np.ndarray] = []
    cur_buf: list[np.ndarray] = []
    cur_len = 0
    sep_pause = np.zeros(int(TARGET_SR * 0.04), dtype=np.float32)

    for w in trimmed:
        cur_buf.append(w)
        cur_buf.append(sep_pause)
        cur_len += w.size + sep_pause.size
        if cur_len >= chunk_samples_limit:
            macro_chunks.append(np.concatenate(cur_buf, axis=0))
            cur_buf = []
            cur_len = 0
        if (len(macro_chunks) * 18 * 50) >= max_frames + 200:
            break

    if cur_buf and (len(macro_chunks) * 18 * 50) < max_frames:
        macro_chunks.append(np.concatenate(cur_buf, axis=0))

    feat_blocks: list[torch.Tensor] = []
    total_f = 0
    for mc in macro_chunks:
        f = vc.extract_wavlm_features(mc, sample_rate=TARGET_SR, trim_silence=False).cpu()
        feat_blocks.append(f)
        total_f += int(f.shape[0])
        if total_f >= max_frames:
            break

    pool = torch.cat(feat_blocks, dim=0)
    if pool.shape[0] > max_frames:
        pool = pool[:max_frames]
    return pool


def build_target_pool_cache(
    pool_dir: str | Path = TARGET_POOL_DIR,
    output_cache_path: str | Path = MODELS_CACHE_DIR / "pool_cache.pt",
    vc_engine: NeuralVCEngine | None = None,
    biometrics_engine: SpeakerBiometricsEngine | None = None,
    max_frames_per_speaker: int = 5200,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    pool_path = Path(pool_dir)
    out_path = Path(output_cache_path)

    meta_file = pool_path / "metadata.json"
    pool_meta_map: dict[str, dict[str, Any]] = {}
    dataset_source = "Multi-Corpus Real Vietnamese Speech"
    corpora_included: list[str] = []
    if meta_file.exists():
        meta_json = json.loads(meta_file.read_text(encoding="utf-8"))
        dataset_source = meta_json.get("dataset_source", dataset_source)
        corpora_included = meta_json.get("corpora_included", [])
        for spk_info in meta_json.get("speakers", []):
            pool_meta_map[spk_info["speaker_id"]] = spk_info

    speaker_dirs = sorted([d for d in pool_path.iterdir() if d.is_dir()])
    if not speaker_dirs:
        raise RuntimeError(f"Không tìm thấy thư mục speaker nào trong '{pool_path}'.")

    vc = vc_engine or NeuralVCEngine()
    bio = biometrics_engine or SpeakerBiometricsEngine(device=vc.device)

    speakers_cache: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    print(
        f"[ReVox Pool Cache V2] Đang trích xuất đám mây đặc trưng mật độ cao ({max_frames_per_speaker} frames/speaker) "
        f"cho {len(speaker_dirs)} speakers trên '{vc.device}'..."
    )

    for spk_dir in tqdm(speaker_dirs, desc="Building Dense Pool"):
        spk_id = spk_dir.name
        wav_files = sorted(spk_dir.glob("*.wav"))
        if not wav_files:
            continue

        wavs: list[np.ndarray] = []
        total_dur = 0.0
        for wf in wav_files:
            w = load_mono_16k(wf, sr=TARGET_SR)
            if w.size >= TARGET_SR // 2:
                wavs.append(w)
                total_dur += float(w.size) / float(TARGET_SR)

        if not wavs:
            continue

        # Extract ECAPA-TDNN voiceprint from 8 representative utterances (~30s speech)
        utts_embs = [bio.extract_embedding(w, sample_rate=TARGET_SR) for w in wavs[:8]]
        mean_emb = np.mean(np.stack(utts_embs, axis=0), axis=0)
        mean_emb = (mean_emb / (np.linalg.norm(mean_emb) + 1e-8)).astype(np.float32)

        wavlm_pool = _extract_chunked_wavlm_pool(vc, wavs, max_frames=max_frames_per_speaker)
        wavlm_pool_fp16 = wavlm_pool.to(torch.float16)

        concat_for_f0 = np.concatenate(wavs[:3], axis=0)
        inferred_gender, median_f0 = estimate_gender_f0(concat_for_f0, sample_rate=TARGET_SR)
        known_meta = pool_meta_map.get(spk_id, {})
        gender = str(known_meta.get("gender") or inferred_gender).lower()
        if gender not in ("male", "female"):
            gender = inferred_gender

        raw_smooth, raw_warm = _compute_acoustic_warmth_and_smoothness(wavs, gender, median_f0)

        speakers_cache.append(
            {
                "speaker_id": spk_id,
                "gender": gender,
                "corpus": known_meta.get("corpus", "VIVOS"),
                "median_f0_hz": round(float(known_meta.get("median_f0_hz", median_f0)), 2),
                "num_utterances": len(wavs),
                "total_duration_sec": round(total_dur, 2),
                "num_wavlm_frames": int(wavlm_pool_fp16.shape[0]),
                "spectral_smoothness": round(raw_smooth, 4),
                "warmth_score": round(raw_warm, 4),
                "ecapa_embedding": mean_emb,
                "wavlm_pool": wavlm_pool_fp16,
            }
        )

    if len(speakers_cache) > 1:
        sm_vals = np.array([e["spectral_smoothness"] for e in speakers_cache], dtype=np.float64)
        wm_vals = np.array([e["warmth_score"] for e in speakers_cache], dtype=np.float64)
        sm_min, sm_ptp = float(sm_vals.min()), max(float(np.ptp(sm_vals)), 1e-6)
        wm_min, wm_ptp = float(wm_vals.min()), max(float(np.ptp(wm_vals)), 1e-6)
        for e in speakers_cache:
            e["spectral_smoothness"] = round(0.60 + 0.38 * ((e["spectral_smoothness"] - sm_min) / sm_ptp), 4)
            e["warmth_score"] = round(0.60 + 0.38 * ((e["warmth_score"] - wm_min) / wm_ptp), 4)

    for e in speakers_cache:
        summary_records.append({k: v for k, v in e.items() if k not in ("ecapa_embedding", "wavlm_pool")})

    elapsed = round(time.perf_counter() - t_start, 2)
    male_count = sum(1 for s in summary_records if s["gender"] == "male")
    female_count = sum(1 for s in summary_records if s["gender"] == "female")

    cache_payload = {
        "version": 2,
        "dataset_source": dataset_source,
        "corpora_included": corpora_included,
        "num_speakers": len(speakers_cache),
        "male_speakers": male_count,
        "female_speakers": female_count,
        "build_seconds": elapsed,
        "speakers": speakers_cache,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, str(out_path))

    summary_json_path = pool_path / "pool_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "dataset_source": dataset_source,
                "corpora_included": corpora_included,
                "num_speakers": len(summary_records),
                "male_speakers": male_count,
                "female_speakers": female_count,
                "total_utterances": sum(s["num_utterances"] for s in summary_records),
                "total_duration_minutes": round(sum(s["total_duration_sec"] for s in summary_records) / 60.0, 2),
                "build_seconds": elapsed,
                "cache_file": str(out_path),
                "cache_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
                "speakers": summary_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[ReVox Pool Cache V2] Hoàn tất! {len(speakers_cache)} speakers "
        f"({male_count} Nam, {female_count} Nữ), cache={out_path.stat().st_size / 1e6:.1f}MB ({elapsed}s)."
    )
    return cache_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense WavLM + ECAPA cache for ReVox target_pool.")
    parser.add_argument("--pool-dir", type=str, default=str(TARGET_POOL_DIR))
    parser.add_argument("--output", type=str, default=str(MODELS_CACHE_DIR / "pool_cache.pt"))
    parser.add_argument("--max-frames", type=int, default=5200)
    args = parser.parse_args()
    build_target_pool_cache(
        pool_dir=args.pool_dir,
        output_cache_path=args.output,
        max_frames_per_speaker=args.max_frames,
    )


if __name__ == "__main__":
    main()
