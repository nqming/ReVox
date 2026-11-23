"""
Comprehensive Scientific Benchmark & Visualization Suite for ReVox (`scripts/evaluate_revox.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revox import OUTPUTS_DIR, TEST_SAMPLES_DIR  # noqa: E402
from revox.domain_selector import DOMAIN_PROFILES, select_target_speaker  # noqa: E402
from revox.evaluator import (  # noqa: E402
    compute_cer,
    compute_eer,
    compute_wer,
    cosine_similarity,
    plot_mel_trio,
)
from revox.pipeline import ReVoxPipeline  # noqa: E402
from revox.speaker_biometrics import TARGET_SR, load_mono_16k  # noqa: E402


def _plot_similarity_distributions(
    impostor_scores: list[float],
    anon_vs_orig_scores: list[float],
    recon_vs_orig_scores: list[float],
    genuine_orig_scores: list[float],
    eer_threshold: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=160)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bins = np.linspace(-0.20, 1.0, 40)
    series = [
        (impostor_scores, "Khác người nói (Impostor)", "#64748b", 0.40),
        (anon_vs_orig_scores, "Đã Ẩn Danh vs Gốc", "#ef4444", 0.58),
        (recon_vs_orig_scores, "Đã Khôi Phục vs Gốc", "#10b981", 0.62),
        (genuine_orig_scores, "Cùng người nói gốc", "#3b82f6", 0.48),
    ]

    for values, label, color, alpha in series:
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=alpha,
            color=color,
            edgecolor=color,
            linewidth=1.1,
            label=f"{label} (μ={arr.mean():.3f})",
        )

    ax.axvline(
        eer_threshold,
        color="#facc15",
        linestyle="--",
        linewidth=1.8,
        label=f"Ngưỡng ECAPA θ_EER = {eer_threshold:.3f}",
    )

    ax.set_title(
        "Phân phối Điểm Tương Đồng Sinh Trắc Học ECAPA-TDNN",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("ECAPA-TDNN Cosine Similarity", color="#e2e8f0", fontsize=10)
    ax.set_ylabel("Mật độ", color="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.25, color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    leg = ax.legend(facecolor="#0f172a", edgecolor="#334155", fontsize=9, loc="upper left")
    for text in leg.get_texts():
        text.set_color("#f1f5f9")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_eer_and_wer_summary(summary_metrics: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=160)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.25, color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    eer_labels = ["Giọng Gốc", "Ẩn danh\nNgân hàng", "Ẩn danh\nY tế", "Ẩn danh\nNhân chứng", "Khôi phục"]
    eer_vals = [
        summary_metrics["eer_original_pct"],
        summary_metrics["profiles"]["banking_call_center"]["eer_anon_pct"],
        summary_metrics["profiles"]["medical_telehealth"]["eer_anon_pct"],
        summary_metrics["profiles"]["whistleblower_max"]["eer_anon_pct"],
        summary_metrics["eer_reidentified_pct"],
    ]
    colors_eer = ["#3b82f6", "#f97316", "#eab308", "#ef4444", "#10b981"]

    bars1 = axes[0].bar(eer_labels, eer_vals, color=colors_eer, width=0.55, edgecolor="#0f172a")
    axes[0].axhline(50.0, color="#94a3b8", linestyle="--", linewidth=1.2, label="Ẩn danh lý tưởng (50%)")
    axes[0].set_ylim(0, max(58.0, max(eer_vals) + 8.0))
    axes[0].set_title("Tỷ lệ lỗi ngang bằng EER (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("EER (%)", color="#e2e8f0")
    leg0 = axes[0].legend(facecolor="#0f172a", edgecolor="#334155", fontsize=8.5)
    for t in leg0.get_texts():
        t.set_color("#f1f5f9")
    for b in bars1:
        h = b.get_height()
        axes[0].text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            color="#f8fafc",
            fontsize=9,
            fontweight="bold",
        )

    stages = ["Giọng Gốc", "Giọng Ẩn Danh", "Giọng Khôi Phục"]
    wer_vals = [
        summary_metrics["wer_original_pct"],
        summary_metrics["wer_anonymized_pct"],
        summary_metrics["wer_reidentified_pct"],
    ]
    cer_vals = [
        summary_metrics["cer_original_pct"],
        summary_metrics["cer_anonymized_pct"],
        summary_metrics["cer_reidentified_pct"],
    ]
    x = np.arange(len(stages))
    width = 0.34
    b_wer = axes[1].bar(x - width / 2, wer_vals, width, label="WER (%) - Lỗi Từ", color="#38bdf8")
    b_cer = axes[1].bar(x + width / 2, cer_vals, width, label="CER (%) - Lỗi Ký Tự", color="#a78bfa")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages)
    axes[1].set_ylim(0, max(max(wer_vals + cer_vals) * 1.28, 15.0))
    axes[1].set_title("Độ rõ chữ Tiếng Việt (Faster-Whisper)", color="#f8fafc", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Tỷ lệ lỗi (%)", color="#e2e8f0")
    leg1 = axes[1].legend(facecolor="#0f172a", edgecolor="#334155", fontsize=8.5)
    for t in leg1.get_texts():
        t.set_color("#f1f5f9")

    for bars in (b_wer, b_cer):
        for b in bars:
            h = b.get_height()
            axes[1].text(
                b.get_x() + b.get_width() / 2.0,
                h + 0.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_evaluation(
    test_dir: str | Path = TEST_SAMPLES_DIR,
    output_dir: str | Path = OUTPUTS_DIR,
    max_trials_per_speaker: int = 2,
    master_password: str = "ReVox-Judicial-Key-2026!",
) -> dict[str, Any]:
    t_global = time.perf_counter()
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    eval_wav_dir = out_path / "eval_samples"
    eval_wav_dir.mkdir(parents=True, exist_ok=True)

    meta_path = test_path / "metadata.json"
    test_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    utterances: list[dict[str, Any]] = test_meta.get("utterances", [])

    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for item in utterances:
        by_speaker.setdefault(item["speaker_id"], []).append(item)

    speakers = sorted(by_speaker.keys())
    print(f"[ReVox Benchmark] Đánh giá trên {len(speakers)} người nói độc lập ({len(utterances)} câu nói thật)...")

    pipe = ReVoxPipeline(load_asr=True, whisper_model_size="small")

    enrollment_embs: dict[str, np.ndarray] = {}
    trial_items: list[dict[str, Any]] = []

    for spk in tqdm(speakers, desc="1/3 Extracting Original Biometrics"):
        spk_utts = by_speaker[spk]
        enroll_utt = spk_utts[0]
        enrollment_embs[spk] = pipe.biometrics.extract_embedding(test_path / enroll_utt["relative_path"])

        trials_for_spk = spk_utts[1 : 1 + max_trials_per_speaker] if len(spk_utts) > 1 else spk_utts[:1]
        for u in trials_for_spk:
            wav_p = test_path / u["relative_path"]
            emb_orig = pipe.biometrics.extract_embedding(wav_p)
            asr_orig = pipe.asr_engine.transcribe(wav_p, language="vi")["text"]
            trial_items.append(
                {
                    **u,
                    "abs_path": wav_p,
                    "orig_emb": emb_orig,
                    "asr_orig": asr_orig,
                    "wer_orig": compute_wer(u["ground_truth_text"], asr_orig),
                    "cer_orig": compute_cer(u["ground_truth_text"], asr_orig),
                }
            )

    genuine_orig: list[float] = []
    impostor_orig: list[float] = []
    for item in trial_items:
        spk = item["speaker_id"]
        emb = item["orig_emb"]
        for enroll_spk, enroll_emb in enrollment_embs.items():
            score = cosine_similarity(emb, enroll_emb)
            if enroll_spk == spk:
                genuine_orig.append(score)
            else:
                impostor_orig.append(score)

    eer_orig_res = compute_eer(genuine_orig, impostor_orig)
    theta_eer = eer_orig_res["threshold"]

    profile_names = list(DOMAIN_PROFILES.keys())
    profile_genuine_anon: dict[str, list[float]] = {p: [] for p in profile_names}
    profile_impostor_anon: dict[str, list[float]] = {p: [] for p in profile_names}
    profile_self_cos: dict[str, list[float]] = {p: [] for p in profile_names}

    detailed_rows: list[dict[str, Any]] = []
    genuine_recon_enroll: list[float] = []
    impostor_recon_enroll: list[float] = []
    recon_self_cos_list: list[float] = []
    anon_self_cos_primary: list[float] = []
    rtf_deid_list: list[float] = []
    rtf_reid_list: list[float] = []
    xrt_deid_list: list[float] = []
    rvox_kb_list: list[float] = []
    wer_anon_list: list[float] = []
    cer_anon_list: list[float] = []
    wer_recon_list: list[float] = []
    cer_recon_list: list[float] = []

    sample_trio_paths: tuple[Path, Path, Path, float, float] | None = None

    for idx, item in enumerate(tqdm(trial_items, desc="2/3 Anonymizing & Re-identifying Trials")):
        spk = item["speaker_id"]
        wav_orig_path: Path = item["abs_path"]
        gt_text = item["ground_truth_text"]

        primary_profile = profile_names[idx % len(profile_names)]
        wav_src_16k = load_mono_16k(wav_orig_path, sr=TARGET_SR)
        z_src_cached = pipe.vc_engine.extract_wavlm_features(wav_src_16k, sample_rate=TARGET_SR, trim_silence=False)
        e_src_cached = item["orig_emb"]
        in_gender = item.get("gender") or "male"

        for prof in profile_names:
            is_primary = prof == primary_profile
            if not is_primary:
                sel = select_target_speaker(
                    input_ecapa_embedding=e_src_cached,
                    input_gender=in_gender,
                    pool_entries=pipe.pool_entries,
                    domain_profile=prof,
                )
                t_pool = sel["target_entry"]["wavlm_pool"].to(device=pipe.device, dtype=torch.float32)
                z_mapped = pipe.vc_engine.knn_convert_features(z_src_cached, t_pool, topk=sel["knn_topk"])
                w_anon_alt = pipe.vc_engine.vocode_features(z_mapped)
                e_anon = pipe.biometrics.extract_embedding(w_anon_alt, sample_rate=TARGET_SR)
                cos_self = cosine_similarity(e_src_cached, e_anon)
                profile_self_cos[prof].append(round(cos_self, 4))
                for enroll_spk, enroll_emb in enrollment_embs.items():
                    sc = cosine_similarity(e_anon, enroll_emb)
                    if enroll_spk == spk:
                        profile_genuine_anon[prof].append(sc)
                    else:
                        profile_impostor_anon[prof].append(sc)
                continue

            out_anon = (
                eval_wav_dir / f"{spk}_{idx:02d}_{prof}_anon.wav"
                if idx < 12
                else (eval_wav_dir / f"_tmp_{prof}.wav")
            )
            deid_res = pipe.deidentify(
                input_audio=wav_orig_path,
                output_wav_path=out_anon,
                master_password=master_password,
                domain_profile=prof,
                embed_rvox=True,
                run_asr=True,
                input_gender=item.get("gender"),
            )
            e_anon = deid_res["anon_embedding"]
            profile_self_cos[prof].append(deid_res["cosine_similarity"])

            for enroll_spk, enroll_emb in enrollment_embs.items():
                sc = cosine_similarity(e_anon, enroll_emb)
                if enroll_spk == spk:
                    profile_genuine_anon[prof].append(sc)
                else:
                    profile_impostor_anon[prof].append(sc)

            rtf_deid_list.append(deid_res["rtf"])
            xrt_deid_list.append(deid_res["xrt"])
            rvox_kb_list.append(deid_res["rvox_chunk_kb"])
            anon_self_cos_primary.append(deid_res["cosine_similarity"])

            asr_anon = deid_res["transcript"]
            w_anon = compute_wer(gt_text, asr_anon)
            c_anon = compute_cer(gt_text, asr_anon)
            wer_anon_list.append(w_anon)
            cer_anon_list.append(c_anon)

            out_recon = (
                eval_wav_dir / f"{spk}_{idx:02d}_{prof}_recon.wav"
                if idx < 12
                else (eval_wav_dir / "_tmp_recon.wav")
            )
            reid_res = pipe.reidentify(
                anon_wav_path=out_anon,
                master_password=master_password,
                output_wav_path=out_recon,
                original_wav_path=wav_orig_path,
                run_asr=True,
            )
            rtf_reid_list.append(reid_res["rtf"])
            cos_orig_recon = float(reid_res["cosine_orig_vs_recon"])
            recon_self_cos_list.append(cos_orig_recon)

            e_recon = reid_res["recon_embedding"]
            for enroll_spk, enroll_emb in enrollment_embs.items():
                sc_r = cosine_similarity(e_recon, enroll_emb)
                if enroll_spk == spk:
                    genuine_recon_enroll.append(sc_r)
                else:
                    impostor_recon_enroll.append(sc_r)

            asr_recon = reid_res["transcript"]
            w_recon = compute_wer(gt_text, asr_recon)
            c_recon = compute_cer(gt_text, asr_recon)
            wer_recon_list.append(w_recon)
            cer_recon_list.append(c_recon)

            if sample_trio_paths is None and prof == "whistleblower_max" and idx < 12:
                sample_trio_paths = (
                    wav_orig_path,
                    out_anon,
                    out_recon,
                    deid_res["cosine_similarity"],
                    cos_orig_recon,
                )

            detailed_rows.append(
                {
                    "trial_idx": idx,
                    "speaker_id": spk,
                    "corpus": item.get("corpus", "VIVOS"),
                    "gender": item.get("gender", ""),
                    "domain_profile": prof,
                    "target_speaker_id": deid_res["target_speaker_id"],
                    "target_gender": deid_res["target_gender"],
                    "cos_orig_vs_anon": deid_res["cosine_similarity"],
                    "cos_orig_vs_recon": cos_orig_recon,
                    "rvox_chunk_kb": deid_res["rvox_chunk_kb"],
                    "wer_orig_pct": round(item["wer_orig"] * 100.0, 2),
                    "wer_anon_pct": round(w_anon * 100.0, 2),
                    "wer_recon_pct": round(w_recon * 100.0, 2),
                    "cer_orig_pct": round(item["cer_orig"] * 100.0, 2),
                    "cer_anon_pct": round(c_anon * 100.0, 2),
                    "cer_recon_pct": round(c_recon * 100.0, 2),
                    "rtf_deid": deid_res["rtf"],
                    "xrt_deid": deid_res["xrt"],
                    "ground_truth_text": gt_text,
                    "asr_anon_text": asr_anon,
                    "asr_recon_text": asr_recon,
                }
            )

    for tmp_f in eval_wav_dir.glob("_tmp_*.wav"):
        tmp_f.unlink(missing_ok=True)

    profiles_summary: dict[str, Any] = {}
    all_gen_anon: list[float] = []
    all_imp_anon: list[float] = []
    for prof in profile_names:
        eer_p = compute_eer(profile_genuine_anon[prof], profile_impostor_anon[prof])
        all_gen_anon.extend(profile_genuine_anon[prof])
        all_imp_anon.extend(profile_impostor_anon[prof])
        profiles_summary[prof] = {
            "display_name": DOMAIN_PROFILES[prof].display_name,
            "eer_anon_pct": eer_p["eer_percent"],
            "mean_cos_orig_vs_anon": round(float(np.mean(profile_self_cos[prof])), 4),
            "std_cos_orig_vs_anon": round(float(np.std(profile_self_cos[prof])), 4),
        }

    eer_anon_overall = compute_eer(all_gen_anon, all_imp_anon)
    eer_recon_res = compute_eer(genuine_recon_enroll, impostor_recon_enroll)

    summary_metrics = {
        "dataset_source": test_meta.get("dataset_source", "Multi-Corpus Vietnamese Speech"),
        "corpora_included": test_meta.get("corpora_included", ["VIVOS"]),
        "num_test_speakers": len(speakers),
        "num_trial_utterances": len(trial_items),
        "eer_original_pct": eer_orig_res["eer_percent"],
        "eer_threshold": theta_eer,
        "eer_anonymized_overall_pct": eer_anon_overall["eer_percent"],
        "eer_reidentified_pct": eer_recon_res["eer_percent"],
        "mean_cos_genuine_enrollment": round(float(np.mean(genuine_orig)), 4),
        "mean_cos_impostor": round(float(np.mean(impostor_orig)), 4),
        "mean_cos_orig_vs_anon": round(float(np.mean(anon_self_cos_primary)), 4),
        "mean_cos_orig_vs_recon": round(float(np.mean(recon_self_cos_list)), 4),
        "wer_original_pct": round(float(np.mean([x["wer_orig"] for x in trial_items]) * 100.0), 2),
        "wer_anonymized_pct": round(float(np.mean(wer_anon_list) * 100.0), 2),
        "wer_reidentified_pct": round(float(np.mean(wer_recon_list) * 100.0), 2),
        "cer_original_pct": round(float(np.mean([x["cer_orig"] for x in trial_items]) * 100.0), 2),
        "cer_anonymized_pct": round(float(np.mean(cer_anon_list) * 100.0), 2),
        "cer_reidentified_pct": round(float(np.mean(cer_recon_list) * 100.0), 2),
        "mean_rtf_deidentify": round(float(np.mean(rtf_deid_list)), 4),
        "mean_xrt_deidentify": round(float(np.mean(xrt_deid_list)), 2),
        "mean_rtf_reidentify": round(float(np.mean(rtf_reid_list)), 4),
        "mean_rvox_chunk_kb": round(float(np.mean(rvox_kb_list)), 2),
        "profiles": profiles_summary,
        "total_benchmark_sec": round(time.perf_counter() - t_global, 2),
    }

    pd.DataFrame(detailed_rows).to_csv(out_path / "benchmark_results.csv", index=False, encoding="utf-8-sig")
    (out_path / "benchmark_summary.json").write_text(
        json.dumps(summary_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _plot_similarity_distributions(
        impostor_scores=impostor_orig,
        anon_vs_orig_scores=anon_self_cos_primary,
        recon_vs_orig_scores=recon_self_cos_list,
        genuine_orig_scores=genuine_orig,
        eer_threshold=theta_eer,
        out_path=out_path / "similarity_distribution.png",
    )
    _plot_eer_and_wer_summary(summary_metrics, out_path / "eer_wer_benchmark.png")

    if sample_trio_paths is not None:
        orig_p, anon_p, recon_p, c_anon, c_rec = sample_trio_paths
        fig_trio = plot_mel_trio(
            anon_audio=anon_p, recon_audio=recon_p, orig_audio=orig_p, sr=TARGET_SR, anon_cos=c_anon, recon_cos=c_rec
        )
        fig_trio.savefig(str(out_path / "sample_mel_trio.png"), dpi=160, facecolor=fig_trio.get_facecolor())
        plt.close(fig_trio)

    md_lines = [
        "# 📊 Báo Cáo Thực Nghiệm Hệ Thống ReVox V2.0",
        "",
        f"- **Nguồn dữ liệu đa tập:** `{summary_metrics['dataset_source']}`",
        f"- **Quy mô kiểm thử:** `{summary_metrics['num_test_speakers']}` người nói độc lập, `{summary_metrics['num_trial_utterances']}` câu kiểm thử",
        f"- **Dung lượng khóa `RVOX`:** `{summary_metrics['mean_rvox_chunk_kb']} KB` (`AES-256-GCM`)",
        "",
        "| Giai đoạn / Cấu hình | EER (%) | Cosine vs Gốc | WER (%) | CER (%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **1. Giọng gốc (Baseline)** | `{summary_metrics['eer_original_pct']}%` | `{summary_metrics['mean_cos_genuine_enrollment']:.4f}` | `{summary_metrics['wer_original_pct']}%` | `{summary_metrics['cer_original_pct']}%` |",
        f"| **2a. Ẩn danh: Ngân hàng** | `{profiles_summary['banking_call_center']['eer_anon_pct']}%` | `{profiles_summary['banking_call_center']['mean_cos_orig_vs_anon']:.4f}` | — | — |",
        f"| **2b. Ẩn danh: Y tế** | `{profiles_summary['medical_telehealth']['eer_anon_pct']}%` | `{profiles_summary['medical_telehealth']['mean_cos_orig_vs_anon']:.4f}` | — | — |",
        f"| **2c. Ẩn danh: Nhân chứng** | `{profiles_summary['whistleblower_max']['eer_anon_pct']}%` | `{profiles_summary['whistleblower_max']['mean_cos_orig_vs_anon']:.4f}` | — | — |",
        f"| **2. Trung bình Đã Ẩn Danh** | **`{summary_metrics['eer_anonymized_overall_pct']}%`** | **`{summary_metrics['mean_cos_orig_vs_anon']:.4f}`** | **`{summary_metrics['wer_anonymized_pct']}%`** | **`{summary_metrics['cer_anonymized_pct']}%`** |",
        f"| **3. Đã Khôi Phục (ReVox Vault)** | **`{summary_metrics['eer_reidentified_pct']}%`** | **`{summary_metrics['mean_cos_orig_vs_recon']:.4f}`** | **`{summary_metrics['wer_reidentified_pct']}%`** | **`{summary_metrics['cer_reidentified_pct']}%`** |",
    ]
    (out_path / "benchmark_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(
        f"[ReVox Benchmark] Xong ({summary_metrics['total_benchmark_sec']}s)! "
        f"EER: {summary_metrics['eer_original_pct']}% -> {summary_metrics['eer_anonymized_overall_pct']}% -> {summary_metrics['eer_reidentified_pct']}% | "
        f"WER: {summary_metrics['wer_original_pct']}% / {summary_metrics['wer_anonymized_pct']}% / {summary_metrics['wer_reidentified_pct']}%"
    )
    return summary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReVox V2.0 on multi-corpus disjoint test speakers.")
    parser.add_argument("--test-dir", type=str, default=str(TEST_SAMPLES_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--max-trials-per-speaker", type=int, default=2)
    args = parser.parse_args()
    run_evaluation(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        max_trials_per_speaker=args.max_trials_per_speaker,
    )


if __name__ == "__main__":
    main()
