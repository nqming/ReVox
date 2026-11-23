"""
Unified End-to-End Voice Anonymization & Identity Recovery Pipeline for ReVox.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from revox import MODELS_CACHE_DIR, OUTPUTS_DIR, TARGET_POOL_DIR
from revox.crypto_wav import (
    ReVoxChunkMissingError,
    decrypt_payload_aes_gcm,
    extract_rvox_chunk,
    save_wav_with_optional_rvox,
)
from revox.domain_selector import DOMAIN_PROFILES, estimate_gender_f0, select_target_speaker
from revox.evaluator import FasterWhisperEvaluator, compute_realtime_metrics
from revox.neural_vc import NeuralVCEngine
from revox.reversible_codec import (
    DEFAULT_NUM_CENTROIDS,
    build_timbre_codebook,
    deserialize_codebook,
    reverse_knn_reconstruct,
    serialize_codebook,
)
from revox.speaker_biometrics import (
    TARGET_SR,
    SpeakerBiometricsEngine,
    biometric_match_percent,
    cosine_similarity,
    get_optimal_device,
    load_mono_16k,
)

DEFAULT_POOL_CACHE_PATH = MODELS_CACHE_DIR / "pool_cache.pt"


class ReVoxPipeline:
    def __init__(
        self,
        device: str | None = None,
        pool_cache_path: str | Path = DEFAULT_POOL_CACHE_PATH,
        load_asr: bool = False,
        whisper_model_size: str = "small",
    ) -> None:
        self.device = get_optimal_device(device)
        self.pool_cache_path = Path(pool_cache_path)
        self.whisper_model_size = whisper_model_size

        self.vc_engine = NeuralVCEngine(device=self.device)
        self.biometrics = SpeakerBiometricsEngine(device=self.device)
        self._asr_engine: FasterWhisperEvaluator | None = None
        if load_asr:
            self._asr_engine = FasterWhisperEvaluator(model_size=whisper_model_size)

        self.pool_entries: list[dict[str, Any]] = []
        self.load_or_build_pool_cache()

    @property
    def asr_engine(self) -> FasterWhisperEvaluator:
        if self._asr_engine is None:
            self._asr_engine = FasterWhisperEvaluator(model_size=self.whisper_model_size)
        return self._asr_engine

    def load_or_build_pool_cache(self, force_rebuild: bool = False) -> list[dict[str, Any]]:
        if self.pool_cache_path.exists() and not force_rebuild:
            raw_cache = torch.load(str(self.pool_cache_path), map_location="cpu", weights_only=False)
            self.pool_entries = raw_cache.get("speakers", [])
            return self.pool_entries

        speaker_dirs = [d for d in sorted(TARGET_POOL_DIR.iterdir()) if d.is_dir()] if TARGET_POOL_DIR.exists() else []
        if speaker_dirs:
            from scripts.build_pool_cache import build_target_pool_cache

            cache_data = build_target_pool_cache(
                pool_dir=TARGET_POOL_DIR,
                output_cache_path=self.pool_cache_path,
                vc_engine=self.vc_engine,
                biometrics_engine=self.biometrics,
            )
            self.pool_entries = cache_data.get("speakers", [])
        return self.pool_entries

    def _infer_gender_hybrid(self, wav_16k: np.ndarray, ecapa_emb: np.ndarray) -> tuple[str, float]:
        gender_f0, median_f0 = estimate_gender_f0(wav_16k, sample_rate=TARGET_SR)
        if 155.0 <= median_f0 <= 175.0 and self.pool_entries:
            male_embs = [e["ecapa_embedding"] for e in self.pool_entries if e.get("gender") == "male"]
            female_embs = [e["ecapa_embedding"] for e in self.pool_entries if e.get("gender") == "female"]
            if male_embs and female_embs:
                sim_m = cosine_similarity(ecapa_emb, np.mean(male_embs, axis=0))
                sim_f = cosine_similarity(ecapa_emb, np.mean(female_embs, axis=0))
                return ("male" if sim_m >= sim_f else "female"), median_f0
        return gender_f0, median_f0

    def deidentify(
        self,
        input_audio: str | Path | np.ndarray,
        output_wav_path: str | Path | None = None,
        master_password: str | None = None,
        domain_profile: str = "banking_call_center",
        embed_rvox: bool | None = None,
        codebook_clusters: int = DEFAULT_NUM_CENTROIDS,
        run_asr: bool = True,
        input_gender: str | None = None,
        exclude_speaker_id: str | None = None,
    ) -> dict[str, Any]:
        if domain_profile not in DOMAIN_PROFILES:
            raise ValueError(f"Unknown domain_profile: {domain_profile}")
        spec = DOMAIN_PROFILES[domain_profile]

        if embed_rvox is None:
            should_embed = spec.default_embed_rvox
        else:
            should_embed = bool(embed_rvox) if spec.allow_disable_rvox else True

        if should_embed and (not master_password or not master_password.strip()):
            raise ValueError("Vui lòng nhập mật khẩu khóa (Master Password).")

        if not self.pool_entries:
            self.load_or_build_pool_cache()
        if not self.pool_entries:
            raise RuntimeError("Chưa có dữ liệu target_pool!")

        t0 = time.perf_counter()
        wav_src = load_mono_16k(input_audio, sr=TARGET_SR)
        audio_dur = float(wav_src.size) / float(TARGET_SR)

        z_src = self.vc_engine.extract_wavlm_features(wav_src, sample_rate=TARGET_SR, trim_silence=False)
        e_src = self.biometrics.extract_embedding(wav_src, sample_rate=TARGET_SR)

        detected_gender, median_f0 = self._infer_gender_hybrid(wav_src, e_src)
        effective_gender = input_gender.lower() if input_gender else detected_gender

        selection = select_target_speaker(
            input_ecapa_embedding=e_src,
            input_gender=effective_gender,
            pool_entries=self.pool_entries,
            domain_profile=domain_profile,
            exclude_speaker_id=exclude_speaker_id,
        )
        target_entry = selection["target_entry"]
        target_pool_feats = target_entry["wavlm_pool"].to(device=self.device, dtype=torch.float32)

        z_anon_mapped = self.vc_engine.knn_convert_features(
            query_features=z_src,
            matching_pool=target_pool_feats,
            topk=selection["knn_topk"],
        )
        wav_anon = self.vc_engine.vocode_features(z_anon_mapped)

        plaintext_cb: bytes | None = None
        actual_centroids = 0
        if should_embed:
            ecapa_b64 = base64.b64encode(e_src.astype(np.float16).tobytes()).decode("ascii")
            codebook = build_timbre_codebook(
                wavlm_features=z_src,
                n_clusters=codebook_clusters,
                metadata={
                    "domain_profile": domain_profile,
                    "source_gender": effective_gender,
                    "source_f0_hz": median_f0,
                    "target_speaker_id": selection["target_speaker_id"],
                    "ecapa_ref_fp16_b64": ecapa_b64,
                },
            )
            actual_centroids = codebook.num_centroids
            plaintext_cb = serialize_codebook(codebook)

        if output_wav_path is None:
            ts = int(time.time() * 1000)
            output_wav_path = OUTPUTS_DIR / f"anon_{domain_profile}_{ts}.wav"

        wav_info = save_wav_with_optional_rvox(
            output_path=output_wav_path,
            waveform=wav_anon,
            sample_rate=TARGET_SR,
            plaintext_rvox_payload=plaintext_cb,
            master_password=master_password if should_embed else None,
        )
        core_proc_sec = time.perf_counter() - t0
        rt_metrics = compute_realtime_metrics(core_proc_sec, audio_dur)

        e_anon = self.biometrics.extract_embedding(wav_anon, sample_rate=TARGET_SR)
        cos_orig_anon = cosine_similarity(e_src, e_anon)

        transcript = ""
        if run_asr:
            transcript = self.asr_engine.transcribe(wav_anon, language="vi")["text"]

        return {
            "output_wav_path": wav_info["output_path"],
            "has_rvox": wav_info["has_rvox"],
            "rvox_chunk_bytes": wav_info["chunk_bytes"],
            "rvox_chunk_kb": wav_info["chunk_kb"],
            "codebook_centroids": actual_centroids,
            "domain_profile": domain_profile,
            "domain_display_name": spec.display_name,
            "input_gender": effective_gender,
            "input_median_f0_hz": median_f0,
            "target_speaker_id": selection["target_speaker_id"],
            "target_gender": selection["target_gender"],
            "selection_reason": selection["selection_reason"],
            "cosine_similarity": round(cos_orig_anon, 4),
            "similarity_percent": biometric_match_percent(cos_orig_anon),
            "transcript": transcript,
            "orig_embedding": e_src,
            "anon_embedding": e_anon,
            "orig_waveform": wav_src,
            "anon_waveform": wav_anon,
            **rt_metrics,
        }

    def reidentify(
        self,
        anon_wav_path: str | Path,
        master_password: str,
        output_wav_path: str | Path | None = None,
        original_wav_path: str | Path | None = None,
        run_asr: bool = False,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        anon_path = Path(anon_wav_path)

        encrypted_blob = extract_rvox_chunk(anon_path)
        if encrypted_blob is None:
            raise ReVoxChunkMissingError(
                f"File '{anon_path.name}' không chứa khóa khôi phục 'RVOX' (đã ẩn danh 1 chiều hoặc bị xóa chunk)."
            )

        decrypted_lzma = decrypt_payload_aes_gcm(encrypted_blob, master_password)
        codebook = deserialize_codebook(decrypted_lzma)

        wav_anon = load_mono_16k(anon_path, sr=TARGET_SR)
        audio_dur = float(wav_anon.size) / float(TARGET_SR)
        z_anon = self.vc_engine.extract_wavlm_features(wav_anon, sample_rate=TARGET_SR, trim_silence=False)

        z_recovered = reverse_knn_reconstruct(
            anon_features=z_anon,
            codebook=codebook,
            topk=4,
            temperature=0.025,
            demean_weight=0.65,
        )
        wav_recon = self.vc_engine.vocode_features(z_recovered)

        if output_wav_path is None:
            ts = int(time.time() * 1000)
            output_wav_path = OUTPUTS_DIR / f"reidentified_{ts}.wav"
        out_path = Path(output_wav_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), wav_recon, TARGET_SR, subtype="PCM_16", format="WAV")

        core_proc_sec = time.perf_counter() - t0
        rt_metrics = compute_realtime_metrics(core_proc_sec, audio_dur)

        e_anon = self.biometrics.extract_embedding(wav_anon, sample_rate=TARGET_SR)
        e_recon = self.biometrics.extract_embedding(wav_recon, sample_rate=TARGET_SR)
        cos_recon_vs_anon = cosine_similarity(e_recon, e_anon)

        wav_orig: np.ndarray | None = None
        cos_orig_vs_anon: float | None = None
        cos_orig_vs_recon: float | None = None
        score_source = "vault_audit"

        if original_wav_path is not None and Path(original_wav_path).exists():
            wav_orig = load_mono_16k(original_wav_path, sample_rate=TARGET_SR) if False else load_mono_16k(original_wav_path, sr=TARGET_SR)
            e_orig = self.biometrics.extract_embedding(wav_orig, sample_rate=TARGET_SR)
            cos_orig_vs_anon = round(cosine_similarity(e_orig, e_anon), 4)
            cos_orig_vs_recon = round(cosine_similarity(e_orig, e_recon), 4)
            score_source = "ground_truth_file"
        elif "ecapa_ref_fp16_b64" in codebook.metadata:
            ref_bytes = base64.b64decode(codebook.metadata["ecapa_ref_fp16_b64"])
            e_ref = np.frombuffer(ref_bytes, dtype=np.float16).astype(np.float32).copy()
            cos_orig_vs_anon = round(cosine_similarity(e_ref, e_anon), 4)
            cos_orig_vs_recon = round(cosine_similarity(e_ref, e_recon), 4)

        transcript = ""
        if run_asr:
            transcript = self.asr_engine.transcribe(wav_recon, language="vi")["text"]

        return {
            "output_wav_path": str(out_path),
            "rvox_chunk_bytes": len(encrypted_blob),
            "rvox_chunk_kb": round(len(encrypted_blob) / 1024.0, 2),
            "codebook_centroids": codebook.num_centroids,
            "codebook_metadata": {k: v for k, v in codebook.metadata.items() if k != "ecapa_ref_fp16_b64"},
            "cosine_recon_vs_anon": round(cos_recon_vs_anon, 4),
            "cosine_orig_vs_anon": cos_orig_vs_anon,
            "cosine_orig_vs_recon": cos_orig_vs_recon,
            "recon_similarity_percent": (
                biometric_match_percent(cos_orig_vs_recon) if cos_orig_vs_recon is not None else None
            ),
            "anon_similarity_percent": (
                biometric_match_percent(cos_orig_vs_anon) if cos_orig_vs_anon is not None else None
            ),
            "score_source": score_source,
            "transcript": transcript,
            "recon_embedding": e_recon,
            "anon_waveform": wav_anon,
            "recon_waveform": wav_recon,
            "orig_waveform": wav_orig,
            **rt_metrics,
        }


_GLOBAL_PIPELINE: ReVoxPipeline | None = None


def get_pipeline(device: str | None = None, load_asr: bool = False) -> ReVoxPipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = ReVoxPipeline(device=device, load_asr=load_asr)
    return _GLOBAL_PIPELINE


def deidentify_audio(
    input_audio: str | Path | np.ndarray,
    output_wav_path: str | Path | None = None,
    master_password: str | None = None,
    domain_profile: str = "banking_call_center",
    embed_rvox: bool | None = None,
    codebook_clusters: int = DEFAULT_NUM_CENTROIDS,
    run_asr: bool = True,
    input_gender: str | None = None,
) -> dict[str, Any]:
    return get_pipeline().deidentify(
        input_audio=input_audio,
        output_wav_path=output_wav_path,
        master_password=master_password,
        domain_profile=domain_profile,
        embed_rvox=embed_rvox,
        codebook_clusters=codebook_clusters,
        run_asr=run_asr,
        input_gender=input_gender,
    )


def reidentify_audio(
    anon_wav_path: str | Path,
    master_password: str,
    output_wav_path: str | Path | None = None,
    original_wav_path: str | Path | None = None,
    run_asr: bool = False,
) -> dict[str, Any]:
    return get_pipeline().reidentify(
        anon_wav_path=anon_wav_path,
        master_password=master_password,
        output_wav_path=output_wav_path,
        original_wav_path=original_wav_path,
        run_asr=run_asr,
    )
