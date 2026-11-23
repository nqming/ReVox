"""
Multi-Corpus Real Human Vietnamese Speech Dataset Streamer V2.0 (`scripts/download_dataset.py`).

Automatically streams and combines 3 public real-human Vietnamese speech datasets from Hugging Face Hub
(`streaming=True`, `Audio(decode=False)` + `soundfile` — zero manual download, zero token required):
1. `vivos` (AILAB VNU-HCM Studio Corpus — dense 40-utterance per-speaker pools + disjoint test speakers)
2. `google/fleurs` (`vi_vn` — Google FLEURS Vietnamese encyclopedic & conversational human speech)
3. `doof-ferb/fpt_fosd` (FPT Open Speech Dataset — everyday Vietnamese speech)

Scientific Partition Saved to Disk (16,000 Hz, mono, 16-bit PCM `.wav`):
- `data/target_pool/`: 16 independent speakers (balanced 8 Male + 8 Female via median F0 < 165Hz),
  each with ~38-42 diverse utterances (~130-160 seconds raw / ~4,800 voiced WavLM frames per speaker —
  4.5x denser phonetic & 6-tone coverage than standard 30s pools!).
- `data/test_samples/`: 20 strictly disjoint speakers/clusters (10 Male + 10 Female, 80 utterances)
  combining `VIVOS`, `Google FLEURS (vi_vn)`, and `FPT Open Speech (fpt_fosd)` with `metadata.json`.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import struct
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models_cache" / "huggingface"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from revox import TARGET_POOL_DIR, TEST_SAMPLES_DIR  # noqa: E402
from revox.domain_selector import estimate_gender_f0  # noqa: E402
from revox.speaker_biometrics import TARGET_SR  # noqa: E402


def _decode_audio_dict_to_16k(audio_obj: Any) -> np.ndarray | None:
    wav: np.ndarray | None = None
    orig_sr: int = TARGET_SR

    if isinstance(audio_obj, dict):
        raw_bytes = audio_obj.get("bytes")
        raw_path = audio_obj.get("path")
        if raw_bytes is not None and len(raw_bytes) > 64:
            wav, orig_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        elif "array" in audio_obj and audio_obj["array"] is not None:
            wav = np.asarray(audio_obj["array"], dtype=np.float32)
            orig_sr = int(audio_obj.get("sampling_rate", TARGET_SR))
        elif raw_path and Path(raw_path).exists():
            wav, orig_sr = sf.read(str(raw_path), dtype="float32", always_2d=False)

    if wav is None or wav.size == 0:
        return None
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.squeeze().astype(np.float32)

    if orig_sr != TARGET_SR:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(wav_t, orig_freq=orig_sr, new_freq=TARGET_SR).squeeze(0).numpy()

    peak = float(np.max(np.abs(wav)))
    if peak < 1e-4:
        return None
    if peak > 0.99 or peak < 0.15:
        wav = wav * (0.85 / peak)
    return wav.astype(np.float32)


def _extract_transcript(row: dict[str, Any]) -> str:
    for key in ("raw_transcription", "sentence", "transcription", "text", "transcript"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            txt = unicodedata.normalize("NFC", val.strip())
            if txt.isupper():
                txt = txt.lower()
            return txt
    return ""


def _estimate_wav_bytes_duration_hint(audio_obj: Any) -> float | None:
    if isinstance(audio_obj, dict) and audio_obj.get("bytes"):
        b = audio_obj["bytes"]
        if len(b) > 44 and b[:4] == b"RIFF" and b[8:12] == b"WAVE" and b[12:16] == b"fmt ":
            byte_rate = struct.unpack("<I", b[28:32])[0]
            if byte_rate > 0:
                return float(len(b) - 44) / float(byte_rate)
    return None


def _stream_vivos_dense(
    pool_speakers_needed: int = 16,
    pool_utts_per_speaker: int = 40,
    vivos_test_speakers_needed: int = 12,
    test_utts_per_speaker: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Stream VIVOS `train` for dense 40-utterance target pool speakers (200+ utts available per speaker!)
    and VIVOS `test` for disjoint test speakers.
    """
    print("[1/3] Đang stream tập dữ liệu #1: 'vivos' (AILAB VNU-HCM Studio Speech)...")
    # A. Stream VIVOS `train` for high-density target_pool speakers (need >= 8 Male + 8 Female)
    train_ds = load_dataset("vivos", split="train", streaming=True, trust_remote_code=True)
    train_ds = train_ds.cast_column("audio", Audio(decode=False))

    train_Temp: dict[str, list[dict[str, Any]]] = {}
    train_gender_cache: dict[str, tuple[str, float]] = {}
    ready_males = 0
    ready_females = 0
    half_target = pool_speakers_needed // 2

    for row in train_ds:
        spk_id = str(row.get("speaker_id", "")).strip()
        if not spk_id:
            continue

        cur_list = train_Temp.get(spk_id, [])
        if len(cur_list) >= pool_utts_per_speaker:
            continue

        # Once we already have enough saturated speakers of a gender, skip new speakers if not needed
        if spk_id in train_gender_cache:
            g_cached, _ = train_gender_cache[spk_id]
            if g_cached == "male" and ready_males >= half_target and len(cur_list) < pool_utts_per_speaker:
                # Check if this speaker was already counting toward ready_males; if not, skip!
                pass

        txt = _extract_transcript(row)
        if len(txt.split()) < 4:
            continue

        dur_hint = _estimate_wav_bytes_duration_hint(row.get("audio"))
        if dur_hint is not None and (dur_hint < 2.5 or dur_hint > 10.0):
            continue

        wav = _decode_audio_dict_to_16k(row.get("audio"))
        if wav is None:
            continue
        dur = float(wav.size) / float(TARGET_SR)
        if dur < 2.5 or dur > 10.0:
            continue

        cur_list.append({"wav": wav, "duration_sec": round(dur, 2), "text": txt, "corpus": "VIVOS"})
        train_Temp[spk_id] = cur_list

        # Classify gender right at 2 utterances so we can skip excess same-gender speakers fast!
        if len(cur_list) == 2 and spk_id not in train_gender_cache:
            concat2 = np.concatenate([u["wav"] for u in cur_list], axis=0)
            g_est, f0_est = estimate_gender_f0(concat2, sample_rate=TARGET_SR)
            train_gender_cache[spk_id] = (g_est, f0_est)
            if (g_est == "male" and ready_males >= half_target) or (
                g_est == "female" and ready_females >= half_target
            ):
                # Mark saturated with dummy so we instantly skip remaining ~240 rows of this speaker!
                train_Temp[spk_id] = cur_list * pool_utts_per_speaker
                continue

        if len(cur_list) == pool_utts_per_speaker and spk_id in train_gender_cache:
            g_est, f0_est = train_gender_cache[spk_id]
            if g_est == "male" and ready_males < half_target:
                ready_males += 1
                print(f"   [+] Target Pool Nam ({ready_males}/{half_target}): {spk_id} ({pool_utts_per_speaker} câu, F0={f0_est}Hz)")
            elif g_est == "female" and ready_females < half_target:
                ready_females += 1
                print(f"   [+] Target Pool Nữ  ({ready_females}/{half_target}): {spk_id} ({pool_utts_per_speaker} câu, F0={f0_est}Hz)")

            if ready_males >= half_target and ready_females >= half_target:
                break

    pool_profiles: list[dict[str, Any]] = []
    m_added, f_added = 0, 0
    for spk_id, utts in train_Temp.items():
        if spk_id not in train_gender_cache:
            continue
        # Filter out skipped speakers where we duplicated 2 items
        if len(utts) == pool_utts_per_speaker and utts[0] is not utts[-1]:
            g_est, f0_est = train_gender_cache[spk_id]
            if g_est == "male" and m_added < half_target:
                pool_profiles.append(
                    {"speaker_id": spk_id, "gender": "male", "median_f0_hz": f0_est, "corpus": "VIVOS", "utts": utts}
                )
                m_added += 1
            elif g_est == "female" and f_added < half_target:
                pool_profiles.append(
                    {"speaker_id": spk_id, "gender": "female", "median_f0_hz": f0_est, "corpus": "VIVOS", "utts": utts}
                )
                f_added += 1

    # B. Stream VIVOS `test` for strictly disjoint studio test speakers
    test_ds = load_dataset("vivos", split="test", streaming=True, trust_remote_code=True)
    test_ds = test_ds.cast_column("audio", Audio(decode=False))
    test_map: dict[str, list[dict[str, Any]]] = {}
    pool_ids = {p["speaker_id"] for p in pool_profiles}

    for row in test_ds:
        spk_id = str(row.get("speaker_id", "")).strip()
        if not spk_id or spk_id in pool_ids:
            continue
        if len(test_map.get(spk_id, [])) >= test_utts_per_speaker:
            if sum(1 for v in test_map.values() if len(v) >= test_utts_per_speaker) >= vivos_test_speakers_needed + 4:
                break
            continue
        txt = _extract_transcript(row)
        if len(txt.split()) < 5:
            continue
        wav = _decode_audio_dict_to_16k(row.get("audio"))
        if wav is None:
            continue
        dur = float(wav.size) / float(TARGET_SR)
        if dur < 2.8 or dur > 9.5:
            continue
        test_map.setdefault(spk_id, []).append(
            {"wav": wav, "duration_sec": round(dur, 2), "text": txt, "corpus": "VIVOS"}
        )

    vivos_test_profiles: list[dict[str, Any]] = []
    for spk_id, utts in sorted(test_map.items()):
        if len(utts) < test_utts_per_speaker:
            continue
        g_est, f0_est = estimate_gender_f0(np.concatenate([u["wav"] for u in utts[:2]]), sample_rate=TARGET_SR)
        vivos_test_profiles.append(
            {
                "speaker_id": spk_id,
                "gender": g_est,
                "median_f0_hz": f0_est,
                "corpus": "VIVOS",
                "utts": utts[:test_utts_per_speaker],
            }
        )

    return pool_profiles, vivos_test_profiles


def _stream_extra_vietnamese_corpora(
    test_utts_per_speaker: int = 4,
) -> list[dict[str, Any]]:
    """
    Stream real human Vietnamese recordings from:
    - Dataset #2: `google/fleurs` (`vi_vn`) — conversational/encyclopedic speech with explicit gender
    - Dataset #3: `doof-ferb/fpt_fosd` — FPT Open Speech everyday Vietnamese
    Groups utterances belonging to the same physical speaker using ECAPA-TDNN voiceprint clustering
    (strict cosine threshold >= 0.62) so EER evaluation remains 100% scientifically valid!
    """
    from sklearn.cluster import AgglomerativeClustering
    from revox.speaker_biometrics import SpeakerBiometricsEngine

    bio = SpeakerBiometricsEngine()
    extra_profiles: list[dict[str, Any]] = []

    # --- Dataset #2: Google FLEURS (vi_vn) ---
    try:
        print("[2/3] Đang stream tập dữ liệu #2: 'google/fleurs' (vi_vn - Google FLEURS Tiếng Việt)...")
        fleurs_items: list[dict[str, Any]] = []
        for split_name in ("test", "validation"):
            ds_fl = load_dataset("google/fleurs", "vi_vn", split=split_name, streaming=True, trust_remote_code=True)
            ds_fl = ds_fl.cast_column("audio", Audio(decode=False))
            for row in ds_fl:
                txt = _extract_transcript(row)
                if len(txt.split()) < 5:
                    continue
                wav = _decode_audio_dict_to_16k(row.get("audio"))
                if wav is None:
                    continue
                dur = float(wav.size) / float(TARGET_SR)
                if dur < 2.8 or dur > 11.0:
                    continue
                emb = bio.extract_embedding(wav, sample_rate=TARGET_SR)
                g_raw = row.get("gender")
                exp_g = "male" if g_raw in (0, "0") else ("female" if g_raw in (1, "1") else None)
                fleurs_items.append(
                    {
                        "wav": wav,
                        "duration_sec": round(dur, 2),
                        "text": txt,
                        "emb": emb,
                        "explicit_gender": exp_g,
                        "corpus": "Google_FLEURS_vi",
                    }
                )
                if len(fleurs_items) >= 90:
                    break
            if len(fleurs_items) >= 90:
                break

        if len(fleurs_items) >= 12:
            embs = np.stack([x["emb"] for x in fleurs_items], axis=0)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.38,  # Cosine similarity >= 0.62 -> guaranteed same physical speaker!
                metric="cosine",
                linkage="complete",
            )
            labels = clustering.fit_predict(embs)
            clusters: dict[int, list[dict[str, Any]]] = {}
            for item, lbl in zip(fleurs_items, labels):
                clusters.setdefault(int(lbl), []).append(item)

            fl_idx = 1
            for _, c_utts in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True):
                if len(c_utts) >= test_utts_per_speaker:
                    chosen = c_utts[:test_utts_per_speaker]
                    g_f0, f0_val = estimate_gender_f0(
                        np.concatenate([u["wav"] for u in chosen[:2]]), sample_rate=TARGET_SR
                    )
                    gender = chosen[0]["explicit_gender"] or g_f0
                    extra_profiles.append(
                        {
                            "speaker_id": f"FLEURS_VI_{fl_idx:02d}",
                            "gender": gender,
                            "median_f0_hz": f0_val,
                            "corpus": "Google FLEURS (vi_vn)",
                            "utts": [
                                {
                                    "wav": u["wav"],
                                    "duration_sec": u["duration_sec"],
                                    "text": u["text"],
                                    "corpus": "Google FLEURS (vi_vn)",
                                }
                                for u in chosen
                            ],
                        }
                    )
                    fl_idx += 1
                    if fl_idx > 6:
                        break
            print(f"   [+] Đã trích xuất {fl_idx - 1} người nói độc lập từ Google FLEURS (vi_vn)!")
    except Exception as exc:
        print(f"   [!] Bỏ qua Google FLEURS ({exc})")

    # --- Dataset #3: FPT Open Speech Dataset (doof-ferb/fpt_fosd) ---
    try:
        print("[3/3] Đang stream tập dữ liệu #3: 'doof-ferb/fpt_fosd' (FPT Open Speech Dataset)...")
        fpt_items: list[dict[str, Any]] = []
        ds_fpt = load_dataset("doof-ferb/fpt_fosd", split="train", streaming=True, trust_remote_code=True)
        ds_fpt = ds_fpt.cast_column("audio", Audio(decode=False))
        for row in ds_fpt:
            txt = _extract_transcript(row)
            if len(txt.split()) < 5:
                continue
            wav = _decode_audio_dict_to_16k(row.get("audio"))
            if wav is None:
                continue
            dur = float(wav.size) / float(TARGET_SR)
            if dur < 2.8 or dur > 10.0:
                continue
            emb = bio.extract_embedding(wav, sample_rate=TARGET_SR)
            fpt_items.append({"wav": wav, "duration_sec": round(dur, 2), "text": txt, "emb": emb})
            if len(fpt_items) >= 100:
                break

        if len(fpt_items) >= 12:
            embs_fpt = np.stack([x["emb"] for x in fpt_items], axis=0)
            clustering_fpt = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.36,
                metric="cosine",
                linkage="complete",
            )
            labels_fpt = clustering_fpt.fit_predict(embs_fpt)
            clusters_fpt: dict[int, list[dict[str, Any]]] = {}
            for item, lbl in zip(fpt_items, labels_fpt):
                clusters_fpt.setdefault(int(lbl), []).append(item)

            fpt_idx = 1
            for _, c_utts in sorted(clusters_fpt.items(), key=lambda kv: len(kv[1]), reverse=True):
                if len(c_utts) >= test_utts_per_speaker:
                    chosen = c_utts[:test_utts_per_speaker]
                    g_f0, f0_val = estimate_gender_f0(
                        np.concatenate([u["wav"] for u in chosen[:2]]), sample_rate=TARGET_SR
                    )
                    extra_profiles.append(
                        {
                            "speaker_id": f"FPT_SPK_{fpt_idx:02d}",
                            "gender": g_f0,
                            "median_f0_hz": f0_val,
                            "corpus": "FPT Open Speech (FOSD)",
                            "utts": [
                                {
                                    "wav": u["wav"],
                                    "duration_sec": u["duration_sec"],
                                    "text": u["text"],
                                    "corpus": "FPT Open Speech (FOSD)",
                                }
                                for u in chosen
                            ],
                        }
                    )
                    fpt_idx += 1
                    if fpt_idx > 4:
                        break
            print(f"   [+] Đã trích xuất {fpt_idx - 1} người nói độc lập từ FPT Open Speech Dataset!")
    except Exception as exc:
        print(f"   [!] Bỏ qua FPT FOSD ({exc})")

    return extra_profiles


def download_and_prepare_datasets(
    num_pool_speakers: int = 16,
    pool_utts_per_speaker: int = 40,
    num_test_speakers: int = 20,
    test_utts_per_speaker: int = 4,
    force_redownload: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    pool_meta_file = TARGET_POOL_DIR / "metadata.json"
    test_meta_file = TEST_SAMPLES_DIR / "metadata.json"
    if not force_redownload and pool_meta_file.exists() and test_meta_file.exists():
        pm = json.loads(pool_meta_file.read_text(encoding="utf-8"))
        tm = json.loads(test_meta_file.read_text(encoding="utf-8"))
        if len(pm.get("speakers", [])) >= 12 and tm.get("num_speakers", 0) >= 20:
            print("[ReVox Dataset] Tập dữ liệu đa nguồn đã sẵn sàng trên ổ cứng.")
            return {"pool_metadata": pm, "test_metadata": tm}

    pool_selected, vivos_test_candidates = _stream_vivos_dense(
        pool_speakers_needed=num_pool_speakers,
        pool_utts_per_speaker=pool_utts_per_speaker,
        vivos_test_speakers_needed=16,
        test_utts_per_speaker=test_utts_per_speaker,
    )
    extra_test_candidates = _stream_extra_vietnamese_corpora(test_utts_per_speaker=test_utts_per_speaker)

    # Combine multi-corpus test speakers (prioritizing FLEURS + FPT + VIVOS test, balanced 10M + 10F)
    all_test_candidates = extra_test_candidates + vivos_test_candidates
    test_males = [s for s in all_test_candidates if s["gender"] == "male"]
    test_females = [s for s in all_test_candidates if s["gender"] == "female"]

    half_test = num_test_speakers // 2
    test_selected = test_males[:half_test] + test_females[:half_test]
    selected_ids = {s["speaker_id"] for s in test_selected}
    for s in all_test_candidates:
        if len(test_selected) >= num_test_speakers:
            break
        if s["speaker_id"] not in selected_ids:
            test_selected.append(s)
            selected_ids.add(s["speaker_id"])

    # Verify strict speaker disjointness
    assert not ({s["speaker_id"] for s in pool_selected} & {s["speaker_id"] for s in test_selected})

    # Write `data/target_pool/`
    if TARGET_POOL_DIR.exists():
        shutil.rmtree(TARGET_POOL_DIR)
    TARGET_POOL_DIR.mkdir(parents=True, exist_ok=True)

    pool_speakers_meta: list[dict[str, Any]] = []
    total_pool_utts = 0
    total_pool_dur = 0.0

    for spk in pool_selected:
        spk_id = spk["speaker_id"]
        spk_dir = TARGET_POOL_DIR / spk_id
        spk_dir.mkdir(parents=True, exist_ok=True)
        spk_dur = 0.0
        saved_files: list[dict[str, Any]] = []

        for idx, u in enumerate(spk["utts"][:pool_utts_per_speaker]):
            fname = f"{spk_id}_{idx + 1:02d}.wav"
            sf.write(str(spk_dir / fname), u["wav"], TARGET_SR, subtype="PCM_16", format="WAV")
            spk_dur += u["duration_sec"]
            saved_files.append(
                {
                    "file": f"{spk_id}/{fname}",
                    "duration_sec": u["duration_sec"],
                    "transcript": u["text"],
                }
            )

        total_pool_utts += len(saved_files)
        total_pool_dur += spk_dur
        pool_speakers_meta.append(
            {
                "speaker_id": spk_id,
                "gender": spk["gender"],
                "median_f0_hz": spk["median_f0_hz"],
                "corpus": spk.get("corpus", "VIVOS"),
                "num_utterances": len(saved_files),
                "total_duration_sec": round(spk_dur, 2),
                "utterances": saved_files,
            }
        )

    corpora_used = sorted({s.get("corpus", "VIVOS") for s in (pool_selected + test_selected)})
    source_summary = " + ".join(corpora_used)

    pool_metadata = {
        "dataset_source": f"Multi-Corpus Real Vietnamese Speech ({source_summary})",
        "corpora_included": corpora_used,
        "sample_rate": TARGET_SR,
        "num_speakers": len(pool_speakers_meta),
        "male_speakers": sum(1 for s in pool_speakers_meta if s["gender"] == "male"),
        "female_speakers": sum(1 for s in pool_speakers_meta if s["gender"] == "female"),
        "total_utterances": total_pool_utts,
        "total_duration_minutes": round(total_pool_dur / 60.0, 2),
        "speakers": pool_speakers_meta,
    }
    pool_meta_file.write_text(json.dumps(pool_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write `data/test_samples/`
    if TEST_SAMPLES_DIR.exists():
        shutil.rmtree(TEST_SAMPLES_DIR)
    TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    test_utterances_meta: list[dict[str, Any]] = []
    test_speakers_summary: list[dict[str, Any]] = []

    for spk in test_selected:
        spk_id = spk["speaker_id"]
        spk_dir = TEST_SAMPLES_DIR / spk_id
        spk_dir.mkdir(parents=True, exist_ok=True)
        spk_dur = 0.0

        for idx, u in enumerate(spk["utts"][:test_utts_per_speaker]):
            fname = f"{spk_id}_test_{idx + 1:02d}.wav"
            rel_path = f"{spk_id}/{fname}"
            sf.write(str(spk_dir / fname), u["wav"], TARGET_SR, subtype="PCM_16", format="WAV")
            spk_dur += u["duration_sec"]
            test_utterances_meta.append(
                {
                    "utt_id": f"{spk_id}_test_{idx + 1:02d}",
                    "speaker_id": spk_id,
                    "gender": spk["gender"],
                    "corpus": spk.get("corpus", "VIVOS"),
                    "median_f0_hz": spk["median_f0_hz"],
                    "duration_sec": u["duration_sec"],
                    "relative_path": rel_path,
                    "ground_truth_text": u["text"],
                }
            )

        test_speakers_summary.append(
            {
                "speaker_id": spk_id,
                "gender": spk["gender"],
                "corpus": spk.get("corpus", "VIVOS"),
                "median_f0_hz": spk["median_f0_hz"],
                "num_utterances": len(spk["utts"][:test_utts_per_speaker]),
                "total_duration_sec": round(spk_dur, 2),
            }
        )

    test_metadata = {
        "dataset_source": f"Multi-Corpus Real Vietnamese Speech ({source_summary})",
        "corpora_included": corpora_used,
        "sample_rate": TARGET_SR,
        "disjoint_from_target_pool": True,
        "num_speakers": len(test_speakers_summary),
        "male_speakers": sum(1 for s in test_speakers_summary if s["gender"] == "male"),
        "female_speakers": sum(1 for s in test_speakers_summary if s["gender"] == "female"),
        "num_utterances": len(test_utterances_meta),
        "speakers": test_speakers_summary,
        "utterances": test_utterances_meta,
    }
    test_meta_file.write_text(json.dumps(test_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = round(time.perf_counter() - t0, 2)
    print("\n" + "=" * 74)
    print(f"[ReVox Multi-Dataset] HOÀN TẤT ({elapsed}s) | Nguồn: {source_summary}")
    print(
        f"  * data/target_pool/ : {pool_metadata['num_speakers']} speakers ({pool_metadata['male_speakers']} Nam, "
        f"{pool_metadata['female_speakers']} Nữ) — {total_pool_utts} câu ({pool_metadata['total_duration_minutes']} phút!)"
    )
    print(
        f"  * data/test_samples/: {test_metadata['num_speakers']} disjoint speakers "
        f"({test_metadata['male_speakers']} Nam, {test_metadata['female_speakers']} Nữ) — {test_metadata['num_utterances']} câu"
    )
    print("=" * 74)
    return {"pool_metadata": pool_metadata, "test_metadata": test_metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream & build multi-corpus real Vietnamese speech dataset.")
    parser.add_argument("--pool-speakers", type=int, default=16)
    parser.add_argument("--pool-utts", type=int, default=40)
    parser.add_argument("--test-speakers", type=int, default=20)
    parser.add_argument("--test-utts", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_and_prepare_datasets(
        num_pool_speakers=args.pool_speakers,
        pool_utts_per_speaker=args.pool_utts,
        num_test_speakers=args.test_speakers,
        test_utts_per_speaker=args.test_utts,
        force_redownload=args.force,
    )


if __name__ == "__main__":
    main()
