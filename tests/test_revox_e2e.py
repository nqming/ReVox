"""
End-to-End & Component Unit Tests for ReVox V2.0 (`tests/test_revox_e2e.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revox import TEST_SAMPLES_DIR  # noqa: E402
from revox.crypto_wav import (  # noqa: E402
    ReVoxAuthError,
    ReVoxChunkMissingError,
    decrypt_payload_aes_gcm,
    extract_rvox_chunk,
    inspect_rvox_chunk,
    save_wav_with_optional_rvox,
)
from revox.evaluator import compute_cer, compute_wer  # noqa: E402
from revox.pipeline import get_pipeline  # noqa: E402
from revox.reversible_codec import (  # noqa: E402
    build_timbre_codebook,
    deserialize_codebook,
    serialize_codebook,
)
from revox.speaker_biometrics import TARGET_SR, cosine_similarity  # noqa: E402


def test_riff_rvox_crypto_container_and_tamper_resistance(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    dummy_audio = (0.25 * np.sin(2 * np.pi * 220.0 * np.arange(16000) / 16000.0)).astype(np.float32)

    for payload_len in (1024, 1025):
        secret_data = rng.bytes(payload_len)
        wav_file = tmp_path / f"test_{payload_len}.wav"
        info = save_wav_with_optional_rvox(
            output_path=wav_file,
            waveform=dummy_audio,
            sample_rate=TARGET_SR,
            plaintext_rvox_payload=secret_data,
            master_password="CorrectHorseBatteryStaple!",
        )
        assert info["has_rvox"] is True
        meta = inspect_rvox_chunk(wav_file)
        assert meta["has_rvox"] is True and meta["valid_magic"] is True

        loaded_wav, sr = sf.read(str(wav_file), dtype="float32")
        assert sr == TARGET_SR and np.max(np.abs(loaded_wav - dummy_audio)) < 1e-3

        enc_blob = extract_rvox_chunk(wav_file)
        assert enc_blob is not None
        assert decrypt_payload_aes_gcm(enc_blob, "CorrectHorseBatteryStaple!") == secret_data

        with pytest.raises(ReVoxAuthError):
            decrypt_payload_aes_gcm(enc_blob, "WrongPassword123")

        tampered = bytearray(enc_blob)
        tampered[-5] ^= 0x01
        with pytest.raises(ReVoxAuthError):
            decrypt_payload_aes_gcm(bytes(tampered), "CorrectHorseBatteryStaple!")


def test_timbre_codebook_compression_roundtrip() -> None:
    rng = np.random.default_rng(42)
    fake_wavlm = rng.standard_normal((240, 1024)).astype(np.float32)
    cb = build_timbre_codebook(fake_wavlm, n_clusters=128, metadata={"speaker": "unit_test"})
    blob = serialize_codebook(cb)
    assert 10_000 < len(blob) < 270_000
    cb_restored = deserialize_codebook(blob)
    assert cb_restored.num_centroids == 128
    assert np.max(np.abs(cb.centroids - cb_restored.centroids)) < 2e-2


def test_closed_loop_real_speech_deidentify_and_reidentify(tmp_path: Path) -> None:
    meta_path = TEST_SAMPLES_DIR / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    utterances = meta["utterances"]
    chosen_utt = next((u for u in utterances if u["duration_sec"] >= 3.5), utterances[0])
    real_wav_path = TEST_SAMPLES_DIR / chosen_utt["relative_path"]
    gt_text = chosen_utt["ground_truth_text"]

    pipe = get_pipeline(load_asr=True)
    e_orig = pipe.biometrics.extract_embedding(real_wav_path)
    orig_asr = pipe.asr_engine.transcribe(real_wav_path, language="vi")["text"]
    cer_orig = compute_cer(gt_text, orig_asr)

    anon_wav_path = tmp_path / "anon_whistleblower.wav"
    recon_wav_path = tmp_path / "recon_whistleblower.wav"
    master_pw = "MasterSecret#ReVox2026"

    deid_res = pipe.deidentify(
        input_audio=real_wav_path,
        output_wav_path=anon_wav_path,
        master_password=master_pw,
        domain_profile="whistleblower_max",
        embed_rvox=True,
        run_asr=True,
        input_gender=chosen_utt["gender"],
    )

    assert anon_wav_path.exists() and deid_res["has_rvox"] is True
    cos_anon = deid_res["cosine_similarity"]
    assert cos_anon < 0.45, f"Expected strong de-identification, got {cos_anon:.4f}"

    with pytest.raises(ReVoxAuthError):
        pipe.reidentify(anon_wav_path=anon_wav_path, master_password="WrongPassword!")

    reid_res = pipe.reidentify(
        anon_wav_path=anon_wav_path,
        master_password=master_pw,
        output_wav_path=recon_wav_path,
        original_wav_path=None,
        run_asr=True,
    )
    e_recon = pipe.biometrics.extract_embedding(recon_wav_path)
    cos_recon = cosine_similarity(e_orig, e_recon)

    cer_anon = compute_cer(gt_text, deid_res["transcript"])
    wer_anon = compute_wer(gt_text, deid_res["transcript"])
    print(f"\n[E2E V2] GT   : {gt_text}")
    print(f"[E2E V2] ORIG : {orig_asr} (CER={cer_orig * 100:.1f}%)")
    print(f"[E2E V2] ANON : {deid_res['transcript']} (WER={wer_anon * 100:.1f}%, CER={cer_anon * 100:.1f}%, Cos={cos_anon:.4f})")
    print(f"[E2E V2] RECON: {reid_res['transcript']} (Cos={cos_recon:.4f})")

    assert cos_recon > cos_anon + 0.20 and cos_recon >= 0.50
    assert cer_anon <= max(0.32, cer_orig + 0.20)

    oneway_path = tmp_path / "anon_oneway.wav"
    oneway_res = pipe.deidentify(
        input_audio=real_wav_path,
        output_wav_path=oneway_path,
        master_password=None,
        domain_profile="whistleblower_max",
        embed_rvox=False,
        run_asr=False,
    )
    assert oneway_res["has_rvox"] is False
    with pytest.raises(ReVoxChunkMissingError):
        pipe.reidentify(anon_wav_path=oneway_path, master_password=master_pw)
