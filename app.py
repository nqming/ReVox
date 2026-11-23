"""
ReVox Studio Web Application (`app.py`) — Minimalist Left-Sidebar Gradio UI.

Navigation (Left Sidebar):
- 🛡️ Ẩn danh giọng nói
- 🔓 Khôi phục danh tính
- 📊 Thống kê & Đánh giá
"""

from __future__ import annotations

import argparse
import html as html_lib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

import gradio as gr
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revox import OUTPUTS_DIR, TARGET_POOL_DIR, TEST_SAMPLES_DIR  # noqa: E402
from revox.crypto_wav import (  # noqa: E402
    ReVoxAuthError,
    ReVoxChunkMissingError,
    ReVoxCryptoError,
    inspect_rvox_chunk,
)
from revox.domain_selector import DOMAIN_PROFILES  # noqa: E402
from revox.evaluator import plot_mel_pair, plot_mel_trio  # noqa: E402
from revox.pipeline import get_pipeline  # noqa: E402

STUDIO_CSS = """
.gradio-container {
    max-width: 1460px !important;
    padding: 18px 22px !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* CSS-based visibility toggle to keep Gradio 6 Svelte DOM mounted cleanly */
.hidden-view {
    display: none !important;
}

/* Left Sidebar */
.revox-sidebar {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    padding: 20px 14px !important;
    min-height: 620px;
}

.revox-brand {
    padding: 4px 10px 18px 10px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 14px;
}
.revox-brand-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.revox-brand-sub {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 4px;
}

.nav-btn {
    justify-content: flex-start !important;
    text-align: left !important;
    min-height: 46px !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    border-radius: 11px !important;
    margin-bottom: 6px !important;
    padding: 0 14px !important;
    white-space: nowrap !important;
    border: 1px solid transparent !important;
}

/* Emerald Primary Styling instead of default Gradio orange */
button.primary {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    border-color: #10b981 !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.22) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
}

/* Prominent Single-Row Source Switcher (Upload / Mic / Sample) */
.source-bar {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
}
.src-btn {
    min-width: 0 !important;
    flex: 1 1 0% !important;
    min-height: 46px !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    border-radius: 11px !important;
    padding: 0 10px !important;
    white-space: nowrap !important;
}

.source-pane {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 8px !important;
}

/* Fix microphone select dropdown clipping inside Gradio Audio */
.source-pane select, .source-pane . microphone-select {
    min-height: 36px !important;
    line-height: 1.3 !important;
}

/* Clean Studio Cards */
.studio-card {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    padding: 22px !important;
}

.section-heading {
    font-size: 1.06rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 12px 0;
    letter-spacing: -0.01em;
}

.cta-btn {
    min-height: 48px !important;
    font-size: 0.98rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    margin-top: 6px !important;
}

/* Compact KPI Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 10px 0;
}
.kpi-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 6px 0 16px 0;
}
.kpi-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 12px 14px;
}
.kpi-label {
    font-size: 0.74rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.kpi-val {
    font-size: 1.25rem;
    font-weight: 800;
    color: #f8fafc;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #34d399;
    margin-top: 3px;
}
.status-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 9px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-pill-muted {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 9px;
    font-size: 0.85rem;
    color: #94a3b8;
    background: #111827;
    border: 1px solid #1f2937;
}
"""


def _resolve_filepath(file_obj: Any) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        p = str(file_obj).strip()
        return p if p and Path(p).exists() else None
    if isinstance(file_obj, dict):
        for k in ("path", "name"):
            if file_obj.get(k) and Path(str(file_obj[k])).exists():
                return str(file_obj[k])
    if hasattr(file_obj, "name") and Path(str(file_obj.name)).exists():
        return str(file_obj.name)
    return None


def _load_test_sample_choices() -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    meta_file = TEST_SAMPLES_DIR / "metadata.json"
    choices: list[tuple[str, str]] = []
    lookup: dict[str, dict[str, Any]] = {}
    if not meta_file.exists():
        return choices, lookup

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        seen_speakers: set[str] = set()
        ordered, remainder = [], []
        for u in meta.get("utterances", []):
            if u["speaker_id"] not in seen_speakers:
                seen_speakers.add(u["speaker_id"])
                ordered.append(u)
            else:
                remainder.append(u)

        for u in (ordered + remainder)[:28]:
            abs_p = str((TEST_SAMPLES_DIR / u["relative_path"]).resolve())
            if not Path(abs_p).exists():
                continue
            g_tag = "Nam" if u.get("gender") == "male" else "Nữ"
            corpus_tag = u.get("corpus", "VIVOS").split()[0]
            preview_txt = u.get("ground_truth_text", "")[:42]
            label = f"{u['speaker_id']} ({g_tag} • {corpus_tag}) — {preview_txt}..."
            choices.append((label, abs_p))
            lookup[abs_p] = u
    except Exception:
        pass
    return choices, lookup


def _list_recent_rvox_wavs() -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    if not OUTPUTS_DIR.exists():
        return results
    candidates = sorted(
        list(OUTPUTS_DIR.glob("anon_*.wav")) + list((OUTPUTS_DIR / "eval_samples").glob("*_anon.wav")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for wav_p in candidates[:12]:
        try:
            info = inspect_rvox_chunk(wav_p)
            tag = f"{info['chunk_kb']} KB" if info["has_rvox"] else "1-chiều"
            results.append((f"{wav_p.name} ({tag})", str(wav_p.resolve())))
        except Exception:
            continue
    return results


def _render_empty_shield_kpis() -> str:
    return """
    <div class="kpi-grid">
        <div class="kpi-box"><div class="kpi-label">Giống giọng gốc</div><div class="kpi-val">—</div><div class="kpi-sub">Chưa xử lý</div></div>
        <div class="kpi-box"><div class="kpi-label">Khóa ngầm RVOX</div><div class="kpi-val">—</div><div class="kpi-sub">AES-256-GCM</div></div>
        <div class="kpi-box"><div class="kpi-label">Thời gian xử lý</div><div class="kpi-val">—</div><div class="kpi-sub">Sẵn sàng</div></div>
    </div>
    """


def _render_empty_vault_kpis() -> str:
    return """
    <div class="kpi-grid">
        <div class="kpi-box"><div class="kpi-label">Khớp giọng gốc</div><div class="kpi-val">—</div><div class="kpi-sub">Chưa giải mã</div></div>
        <div class="kpi-box"><div class="kpi-label">Xác thực AES-GCM</div><div class="kpi-val">—</div><div class="kpi-sub">Chờ mật khẩu</div></div>
        <div class="kpi-box"><div class="kpi-label">Bảng mã âm sắc</div><div class="kpi-val">—</div><div class="kpi-sub">RVOX Chunk</div></div>
    </div>
    """


def run_shield_deidentify(
    source_mode: str,
    upload_audio: str | None,
    mic_audio: str | None,
    sample_audio: str | None,
    domain_profile: str,
    embed_rvox: bool,
    master_password: str,
) -> tuple[str | None, Any, str, str, Any, str | None]:
    if source_mode == "upload":
        active_audio = upload_audio
    elif source_mode == "mic":
        active_audio = mic_audio
    else:
        active_audio = sample_audio

    src_path = _resolve_filepath(active_audio)
    if not src_path:
        raise gr.Error("Chưa có âm thanh đầu vào! Hãy tải file lên, thu âm Mic hoặc chọn một giọng mẫu.")

    spec = DOMAIN_PROFILES.get(domain_profile, DOMAIN_PROFILES["banking_call_center"])
    actual_embed = bool(embed_rvox) if spec.allow_disable_rvox else True

    if actual_embed and (not master_password or not master_password.strip()):
        raise gr.Error("Vui lòng nhập mật khẩu khôi phục (Master Password)!")

    pipe = get_pipeline(load_asr=True)
    res = pipe.deidentify(
        input_audio=src_path,
        master_password=master_password if actual_embed else None,
        domain_profile=domain_profile,
        embed_rvox=actual_embed,
        run_asr=True,
    )

    out_wav = res["output_wav_path"]
    cos_sim = res["cosine_similarity"]
    sim_pct = res["similarity_percent"]
    tg_g = "Nam" if res["target_gender"] == "male" else "Nữ"

    status_txt = "Đã ẩn danh tốt" if cos_sim < 0.35 else f"Đích: {tg_g}"
    rvox_val = f"{res['rvox_chunk_kb']} KB" if res["has_rvox"] else "0 KB"
    rvox_sub = "Đã khóa AES-256" if res["has_rvox"] else "Ẩn danh 1 chiều"

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-box">
            <div class="kpi-label">Giống giọng gốc</div>
            <div class="kpi-val">{sim_pct:.1f}%</div>
            <div class="kpi-sub">{html_lib.escape(status_txt)} (Cos {cos_sim:.2f})</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Khóa ngầm RVOX</div>
            <div class="kpi-val">{rvox_val}</div>
            <div class="kpi-sub">{rvox_sub}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Thời gian xử lý</div>
            <div class="kpi-val">{res['processing_sec']:.2f}s</div>
            <div class="kpi-sub">Đích: {html_lib.escape(res['target_speaker_id'])} ({tg_g})</div>
        </div>
    </div>
    """

    fig = plot_mel_pair(
        orig_audio=res["orig_waveform"],
        anon_audio=res["anon_waveform"],
        cos_sim=cos_sim,
        profile_name=domain_profile,
    )

    return (
        out_wav,
        gr.update(value=out_wav, interactive=True),
        kpi_html,
        res["transcript"] or "(Không phát hiện lời nói)",
        fig,
        out_wav,
    )


def _inspect_vault_file(file_obj: Any) -> tuple[str | None, str]:
    path = _resolve_filepath(file_obj)
    if not path:
        return None, '<span class="status-pill-muted">Chưa chọn file .WAV ẩn danh</span>'
    try:
        info = inspect_rvox_chunk(path)
        fname = html_lib.escape(Path(path).name)
        if info["has_rvox"]:
            return (
                path,
                f'<span class="status-pill">✅ Phát hiện khóa RVOX hợp lệ ({info["chunk_kb"]} KB) — {fname}</span>',
            )
        return (
            path,
            f'<span class="status-pill-muted">⚠️ File {fname} không chứa khóa RVOX (ẩn danh 1 chiều)</span>',
        )
    except Exception as exc:
        return path, f'<span class="status-pill-muted">⚠️ Lỗi đọc file: {html_lib.escape(str(exc))}</span>'


def run_vault_reidentify(
    vault_mode: str,
    uploaded_file: Any,
    recent_path: str | None,
    master_password: str,
    optional_orig_audio: str | None,
) -> tuple[str | None, Any, str, str, Any]:
    candidate = uploaded_file if vault_mode == "upload" else recent_path
    anon_path = _resolve_filepath(candidate)
    if not anon_path:
        raise gr.Error("Vui lòng chọn hoặc tải lên file .WAV đã ẩn danh!")
    if not master_password or not master_password.strip():
        raise gr.Error("Vui lòng nhập mật khẩu khôi phục (Master Password)!")

    orig_path = _resolve_filepath(optional_orig_audio)
    pipe = get_pipeline(load_asr=True)

    try:
        res = pipe.reidentify(
            anon_wav_path=anon_path,
            master_password=master_password,
            original_wav_path=orig_path,
            run_asr=True,
        )
    except (ReVoxAuthError, ReVoxChunkMissingError, ReVoxCryptoError) as err:
        raise gr.Error(str(err)) from err

    cos_anon = res["cosine_orig_vs_anon"]
    cos_recon = res["cosine_orig_vs_recon"]
    recon_pct = res["recon_similarity_percent"] or 0.0
    anon_pct = res["anon_similarity_percent"] or 0.0

    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-box">
            <div class="kpi-label">Khớp giọng gốc</div>
            <div class="kpi-val">{recon_pct:.1f}%</div>
            <div class="kpi-sub">Tăng từ {anon_pct:.1f}% (Cos {cos_recon:.2f})</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Xác thực khóa</div>
            <div class="kpi-val">Hợp lệ</div>
            <div class="kpi-sub">AES-256-GCM Verified</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Bảng mã khôi phục</div>
            <div class="kpi-val">{res['codebook_centroids']} cụm</div>
            <div class="kpi-sub">Khóa RVOX {res['rvox_chunk_kb']} KB</div>
        </div>
    </div>
    """

    fig_trio = plot_mel_trio(
        anon_audio=res["anon_waveform"],
        recon_audio=res["recon_waveform"],
        orig_audio=res["orig_waveform"],
        anon_cos=cos_anon,
        recon_cos=cos_recon,
    )

    out_wav = res["output_wav_path"]
    return (
        out_wav,
        gr.update(value=out_wav, interactive=True),
        kpi_html,
        res["transcript"] or "",
        fig_trio,
    )


def _load_benchmark_dashboard_data() -> tuple[str, pd.DataFrame, pd.DataFrame, str | None, str | None, str | None]:
    pool_meta_p = TARGET_POOL_DIR / "pool_summary.json"
    if not pool_meta_p.exists():
        pool_meta_p = TARGET_POOL_DIR / "metadata.json"
    test_meta_p = TEST_SAMPLES_DIR / "metadata.json"
    bench_json_p = OUTPUTS_DIR / "benchmark_summary.json"

    speakers_rows: list[dict[str, Any]] = []
    pool_spks, test_spks, total_utts = 0, 0, 0
    corpora_str = "VIVOS • FLEURS • FPT"

    if pool_meta_p.exists():
        pm = json.loads(pool_meta_p.read_text(encoding="utf-8"))
        pool_spks = pm.get("num_speakers", 0)
        total_utts += pm.get("total_utterances", pool_spks * 40)
        if pm.get("corpora_included"):
            corpora_str = " • ".join(pm["corpora_included"])
        for s in pm.get("speakers", []):
            speakers_rows.append(
                {
                    "Phân tập": "Bể giọng đích",
                    "Speaker ID": s["speaker_id"],
                    "Nguồn": s.get("corpus", "VIVOS"),
                    "Giới tính": "Nam" if s["gender"] == "male" else "Nữ",
                    "F0 (Hz)": s.get("median_f0_hz", 0),
                    "Số câu": s.get("num_utterances", 0),
                    "Thời lượng (s)": s.get("total_duration_sec", 0),
                }
            )

    if test_meta_p.exists():
        tm = json.loads(test_meta_p.read_text(encoding="utf-8"))
        test_spks = tm.get("num_speakers", 0)
        total_utts += tm.get("num_utterances", 0)
        if tm.get("corpora_included"):
            corpora_str = " • ".join(tm["corpora_included"])
        for s in tm.get("speakers", []):
            speakers_rows.append(
                {
                    "Phân tập": "Kiểm thử độc lập",
                    "Speaker ID": s["speaker_id"],
                    "Nguồn": s.get("corpus", "VIVOS"),
                    "Giới tính": "Nam" if s["gender"] == "male" else "Nữ",
                    "F0 (Hz)": s.get("median_f0_hz", 0),
                    "Số câu": s.get("num_utterances", 0),
                    "Thời lượng (s)": s.get("total_duration_sec", 0),
                }
            )

    eer_orig, eer_anon, eer_rec, cos_rec = "—", "—", "—", "—"
    bench_rows: list[dict[str, Any]] = []

    if bench_json_p.exists():
        bm = json.loads(bench_json_p.read_text(encoding="utf-8"))
        eer_orig = f"{bm['eer_original_pct']}%"
        eer_anon = f"{bm['eer_anonymized_overall_pct']}%"
        eer_rec = f"{bm['eer_reidentified_pct']}%"
        cos_rec = f"{bm['mean_cos_orig_vs_recon'] * 100:.1f}%"
        profs = bm.get("profiles", {})
        bench_rows = [
            {
                "Cấu hình đánh giá": "Giọng thật gốc (Baseline)",
                "EER (%)": f"{bm['eer_original_pct']}%",
                "Cosine vs Gốc": f"{bm['mean_cos_genuine_enrollment']:.3f}",
                "WER (%)": f"{bm['wer_original_pct']}%",
                "CER (%)": f"{bm['cer_original_pct']}%",
            },
            {
                "Cấu hình đánh giá": "Ẩn danh — Tổng đài Ngân hàng",
                "EER (%)": f"{profs.get('banking_call_center', {}).get('eer_anon_pct', '—')}%",
                "Cosine vs Gốc": f"{profs.get('banking_call_center', {}).get('mean_cos_orig_vs_anon', 0):.3f}",
                "WER (%)": "—",
                "CER (%)": "—",
            },
            {
                "Cấu hình đánh giá": "Ẩn danh — Y tế & Tâm lý",
                "EER (%)": f"{profs.get('medical_telehealth', {}).get('eer_anon_pct', '—')}%",
                "Cosine vs Gốc": f"{profs.get('medical_telehealth', {}).get('mean_cos_orig_vs_anon', 0):.3f}",
                "WER (%)": "—",
                "CER (%)": "—",
            },
            {
                "Cấu hình đánh giá": "Ẩn danh — Bảo vệ Nhân chứng",
                "EER (%)": f"{profs.get('whistleblower_max', {}).get('eer_anon_pct', '—')}%",
                "Cosine vs Gốc": f"{profs.get('whistleblower_max', {}).get('mean_cos_orig_vs_anon', 0):.3f}",
                "WER (%)": "—",
                "CER (%)": "—",
            },
            {
                "Cấu hình đánh giá": "Trung bình Đã Ẩn Danh (ReVox Shield)",
                "EER (%)": f"{bm['eer_anonymized_overall_pct']}%",
                "Cosine vs Gốc": f"{bm['mean_cos_orig_vs_anon']:.3f}",
                "WER (%)": f"{bm['wer_anonymized_pct']}%",
                "CER (%)": f"{bm['cer_anonymized_pct']}%",
            },
            {
                "Cấu hình đánh giá": "Đã Khôi Phục Bằng Khóa (ReVox Vault)",
                "EER (%)": f"{bm['eer_reidentified_pct']}%",
                "Cosine vs Gốc": f"{bm['mean_cos_orig_vs_recon']:.3f}",
                "WER (%)": f"{bm['wer_reidentified_pct']}%",
                "CER (%)": f"{bm['cer_reidentified_pct']}%",
            },
        ]

    top_kpis_html = f"""
    <div class="kpi-grid-4">
        <div class="kpi-box">
            <div class="kpi-label">Quy mô Dữ liệu Thật</div>
            <div class="kpi-val">{pool_spks + test_spks} người nói</div>
            <div class="kpi-sub">{total_utts} câu ({html_lib.escape(corpora_str)})</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">EER Giọng Gốc</div>
            <div class="kpi-val">{eer_orig}</div>
            <div class="kpi-sub">Baseline ECAPA-TDNN</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">EER Sau Ẩn Danh</div>
            <div class="kpi-val">{eer_anon}</div>
            <div class="kpi-sub">~50% = Ẩn danh tối đa</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-label">Khôi Phục Danh Tính</div>
            <div class="kpi-val">{cos_rec}</div>
            <div class="kpi-sub">EER giảm về {eer_rec}</div>
        </div>
    </div>
    """

    dist_img = OUTPUTS_DIR / "similarity_distribution.png"
    bar_img = OUTPUTS_DIR / "eer_wer_benchmark.png"
    trio_img = OUTPUTS_DIR / "sample_mel_trio.png"

    return (
        top_kpis_html,
        pd.DataFrame(bench_rows),
        pd.DataFrame(speakers_rows),
        str(dist_img) if dist_img.exists() else None,
        str(bar_img) if bar_img.exists() else None,
        str(trio_img) if trio_img.exists() else None,
    )


def build_revox_app() -> gr.Blocks:
    sample_choices, _ = _load_test_sample_choices()
    default_sample = sample_choices[0][1] if sample_choices else None
    profile_choices = [(spec.display_name, pid) for pid, spec in DOMAIN_PROFILES.items()]

    blocks_kwargs: dict[str, Any] = {"title": "ReVox — Ẩn danh & Khôi phục Giọng nói"}
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kwargs["css"] = STUDIO_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        shield_src_mode = gr.State(value="upload")
        vault_src_mode = gr.State(value="upload")
        last_anon_state = gr.State(value=None)
        last_orig_state = gr.State(value=default_sample)

        with gr.Row():
            # =====================================================================
            # LEFT SIDEBAR NAVIGATION (NO "Tab 1 / Tab 2 / Tab 3" TEXT)
            # =====================================================================
            with gr.Column(scale=1, min_width=245, elem_classes=["revox-sidebar"]):
                gr.HTML(
                    """
                    <div class="revox-brand">
                        <div class="revox-brand-title">🛡️ ReVox</div>
                        <div class="revox-brand-sub">Ẩn danh &amp; Khôi phục Giọng nói</div>
                    </div>
                    """
                )
                nav_shield_btn = gr.Button(
                    "🛡️  Ẩn danh giọng nói", variant="primary", elem_classes=["nav-btn"]
                )
                nav_vault_btn = gr.Button(
                    "🔓  Khôi phục danh tính", variant="secondary", elem_classes=["nav-btn"]
                )
                nav_bench_btn = gr.Button(
                    "📊  Thống kê & Đánh giá", variant="secondary", elem_classes=["nav-btn"]
                )

            # =====================================================================
            # RIGHT MAIN WORKSPACE (MOUNTED CLEANLY VIA CSS CLASSES)
            # =====================================================================
            with gr.Column(scale=5):
                # -----------------------------------------------------------------
                # VIEW 1: ẨN DANH GIỌNG NÓI
                # -----------------------------------------------------------------
                with gr.Column(elem_classes=["view-panel"]) as view_shield:
                    with gr.Row():
                        # Input Card
                        with gr.Column(scale=5, elem_classes=["studio-card"]):
                            gr.HTML('<div class="section-heading">1. Chọn nguồn âm thanh</div>')

                            # 3 Single-Row Prominent Source Switcher Buttons
                            with gr.Row(elem_classes=["source-bar"]):
                                btn_mode_upload = gr.Button(
                                    "📁 Tải file",
                                    variant="primary",
                                    min_width=80,
                                    elem_classes=["src-btn"],
                                )
                                btn_mode_mic = gr.Button(
                                    "🎙️ Thu Mic",
                                    variant="secondary",
                                    min_width=80,
                                    elem_classes=["src-btn"],
                                )
                                btn_mode_sample = gr.Button(
                                    "🎧 Giọng mẫu",
                                    variant="secondary",
                                    min_width=80,
                                    elem_classes=["src-btn"],
                                )

                            # Mode A: File Upload
                            with gr.Column(elem_classes=["source-pane"]) as pane_upload:
                                audio_upload_in = gr.Audio(
                                    sources=["upload"],
                                    type="filepath",
                                    label="Tải file âm thanh (.wav, .mp3)",
                                )

                            # Mode B: Microphone Recording
                            with gr.Column(elem_classes=["source-pane", "hidden-view"]) as pane_mic:
                                audio_mic_in = gr.Audio(
                                    sources=["microphone"],
                                    type="filepath",
                                    label="Thu âm trực tiếp từ Microphone",
                                )

                            # Mode C: Real Human Speech Samples
                            with gr.Column(elem_classes=["source-pane", "hidden-view"]) as pane_sample:
                                sample_dd = gr.Dropdown(
                                    choices=sample_choices,
                                    value=default_sample,
                                    label="Chọn câu nói người thật",
                                )
                                audio_sample_preview = gr.Audio(
                                    value=default_sample,
                                    type="filepath",
                                    label="Nghe giọng mẫu gốc",
                                    interactive=False,
                                )

                            profile_dd = gr.Dropdown(
                                choices=profile_choices,
                                value="banking_call_center",
                                label="Chế độ ẩn danh",
                            )

                            master_pw_in = gr.Textbox(
                                value="ReVox@2026",
                                type="password",
                                label="Mật khẩu khôi phục (Master Password)",
                            )
                            embed_rvox_chk = gr.Checkbox(
                                value=True,
                                elem_classes=["hidden-view"],
                                label="Nhúng khóa khôi phục RVOX",
                            )

                            btn_deidentify = gr.Button(
                                "🛡️ Ẩn danh giọng nói",
                                variant="primary",
                                elem_classes=["cta-btn"],
                            )

                        # Output Card
                        with gr.Column(scale=6, elem_classes=["studio-card"]):
                            gr.HTML('<div class="section-heading">2. Kết quả ẩn danh</div>')

                            anon_audio_out = gr.Audio(
                                type="filepath",
                                label="Giọng đã ẩn danh",
                                interactive=False,
                            )

                            shield_kpi_html = gr.HTML(value=_render_empty_shield_kpis())

                            whisper_asr_box = gr.Textbox(
                                label="Văn bản nhận diện tự động (Faster-Whisper)",
                                interactive=False,
                                lines=2,
                                placeholder="Nội dung lời nói sau khi ẩn danh hiển thị tại đây...",
                            )

                            with gr.Row():
                                btn_download_anon = gr.DownloadButton(
                                    "💾 Tải file .WAV (có khóa)",
                                    variant="secondary",
                                    min_width=120,
                                    interactive=False,
                                )
                                btn_send_to_vault = gr.Button(
                                    "🔓 Khôi phục thử file này →",
                                    variant="secondary",
                                    min_width=120,
                                )

                            with gr.Accordion("📈 Xem phổ âm thanh (Mel-Spectrogram)", open=False):
                                shield_mel_plot = gr.Plot(show_label=False)

                # -----------------------------------------------------------------
                # VIEW 2: KHÔI PHỤC DANH TÍNH
                # -----------------------------------------------------------------
                with gr.Column(elem_classes=["view-panel", "hidden-view"]) as view_vault:
                    with gr.Row():
                        with gr.Column(scale=5, elem_classes=["studio-card"]):
                            gr.HTML('<div class="section-heading">1. Chọn file .WAV đã ẩn danh</div>')

                            with gr.Row(elem_classes=["source-bar"]):
                                btn_vmode_upload = gr.Button(
                                    "📁 Tải file .WAV",
                                    variant="primary",
                                    min_width=100,
                                    elem_classes=["src-btn"],
                                )
                                btn_vmode_recent = gr.Button(
                                    "🕘 File vừa ẩn danh",
                                    variant="secondary",
                                    min_width=100,
                                    elem_classes=["src-btn"],
                                )

                            with gr.Column(elem_classes=["source-pane"]) as vpane_upload:
                                vault_wav_file = gr.File(
                                    file_types=[".wav"],
                                    label="Chọn file .WAV có nhúng khóa RVOX",
                                )

                            with gr.Column(elem_classes=["source-pane", "hidden-view"]) as vpane_recent:
                                recent_anon_dd = gr.Dropdown(
                                    choices=_list_recent_rvox_wavs(),
                                    label="Danh sách file đã ẩn danh gần đây",
                                )

                            vault_status_html = gr.HTML(
                                '<span class="status-pill-muted">Chưa chọn file .WAV ẩn danh</span>'
                            )
                            vault_anon_preview = gr.Audio(
                                type="filepath",
                                label="Nghe file ẩn danh đầu vào",
                                interactive=False,
                            )

                            vault_master_pw = gr.Textbox(
                                value="ReVox@2026",
                                type="password",
                                label="Mật khẩu khôi phục (Master Password)",
                            )

                            with gr.Accordion("➕ Đối chiếu thêm với file gốc (Tùy chọn)", open=False):
                                vault_orig_optional = gr.Audio(
                                    sources=["upload"],
                                    type="filepath",
                                    label="File giọng gốc ban đầu",
                                )

                            btn_reidentify = gr.Button(
                                "🔓 Giải mã & Khôi phục giọng gốc",
                                variant="primary",
                                elem_classes=["cta-btn"],
                            )

                        with gr.Column(scale=6, elem_classes=["studio-card"]):
                            gr.HTML('<div class="section-heading">2. Kết quả khôi phục danh tính</div>')

                            recon_audio_out = gr.Audio(
                                type="filepath",
                                label="Giọng người nói gốc đã khôi phục",
                                interactive=False,
                            )

                            vault_kpi_html = gr.HTML(value=_render_empty_vault_kpis())

                            vault_asr_box = gr.Textbox(
                                label="Văn bản nhận diện trên giọng khôi phục",
                                interactive=False,
                                lines=2,
                            )

                            btn_download_recon = gr.DownloadButton(
                                "💾 Tải file .WAV đã khôi phục",
                                variant="secondary",
                                interactive=False,
                            )

                            with gr.Accordion("📈 Xem đối chiếu 3 phổ âm thanh", open=False):
                                vault_trio_plot = gr.Plot(show_label=False)

                # -----------------------------------------------------------------
                # VIEW 3: THỐNG KÊ & ĐÁNH GIÁ
                # -----------------------------------------------------------------
                with gr.Column(elem_classes=["studio-card", "hidden-view"]) as view_bench:
                    init_kpi, init_df_bench, init_df_spks, init_dist_img, init_bar_img, init_trio_img = (
                        _load_benchmark_dashboard_data()
                    )
                    with gr.Row():
                        gr.HTML('<div class="section-heading">Tổng quan Dữ liệu Đa Nguồn & Kết quả Thực nghiệm</div>')
                        btn_refresh_bench = gr.Button("🔄 Cập nhật", size="sm", min_width=90, scale=0)

                    bench_kpi_html = gr.HTML(value=init_kpi)
                    bench_table = gr.Dataframe(value=init_df_bench, interactive=False, wrap=True)

                    with gr.Row():
                        bench_dist_image = gr.Image(
                            value=init_dist_img, type="filepath", label="Phân phối điểm tương đồng ECAPA-TDNN"
                        )
                        bench_bar_image = gr.Image(
                            value=init_bar_img, type="filepath", label="EER (%) & Độ rõ chữ WER/CER (%)"
                        )

                    with gr.Accordion("👥 Danh sách người nói thật & Phổ đối chiếu mẫu", open=False):
                        bench_trio_image = gr.Image(
                            value=init_trio_img, type="filepath", label="So sánh 3 phổ Mel-Spectrogram"
                        )
                        speakers_table = gr.Dataframe(value=init_df_spks, interactive=False)

        # =====================================================================
        # SIDEBAR NAVIGATION EVENTS (PURE CSS CLASS TOGGLE — ZERO UNMOUNT BUGS)
        # =====================================================================
        def _switch_nav(target: str):
            return (
                gr.update(elem_classes=["view-panel"] if target == "shield" else ["view-panel", "hidden-view"]),
                gr.update(elem_classes=["view-panel"] if target == "vault" else ["view-panel", "hidden-view"]),
                gr.update(elem_classes=["studio-card"] if target == "bench" else ["studio-card", "hidden-view"]),
                gr.update(variant="primary" if target == "shield" else "secondary"),
                gr.update(variant="primary" if target == "vault" else "secondary"),
                gr.update(variant="primary" if target == "bench" else "secondary"),
            )

        nav_outputs = [view_shield, view_vault, view_bench, nav_shield_btn, nav_vault_btn, nav_bench_btn]
        nav_shield_btn.click(fn=lambda: _switch_nav("shield"), inputs=[], outputs=nav_outputs)
        nav_vault_btn.click(fn=lambda: _switch_nav("vault"), inputs=[], outputs=nav_outputs)
        nav_bench_btn.click(fn=lambda: _switch_nav("bench"), inputs=[], outputs=nav_outputs)

        # =====================================================================
        # SHIELD INPUT SOURCE MODE SWITCHER EVENTS
        # =====================================================================
        def _set_shield_source(mode: str):
            return (
                mode,
                gr.update(elem_classes=["source-pane"] if mode == "upload" else ["source-pane", "hidden-view"]),
                gr.update(elem_classes=["source-pane"] if mode == "mic" else ["source-pane", "hidden-view"]),
                gr.update(elem_classes=["source-pane"] if mode == "sample" else ["source-pane", "hidden-view"]),
                gr.update(variant="primary" if mode == "upload" else "secondary"),
                gr.update(variant="primary" if mode == "mic" else "secondary"),
                gr.update(variant="primary" if mode == "sample" else "secondary"),
            )

        src_outputs = [
            shield_src_mode,
            pane_upload,
            pane_mic,
            pane_sample,
            btn_mode_upload,
            btn_mode_mic,
            btn_mode_sample,
        ]
        btn_mode_upload.click(fn=lambda: _set_shield_source("upload"), inputs=[], outputs=src_outputs)
        btn_mode_mic.click(fn=lambda: _set_shield_source("mic"), inputs=[], outputs=src_outputs)
        btn_mode_sample.click(fn=lambda: _set_shield_source("sample"), inputs=[], outputs=src_outputs)

        sample_dd.change(fn=lambda p: (p, p), inputs=[sample_dd], outputs=[audio_sample_preview, last_orig_state])
        audio_upload_in.change(fn=lambda p: p, inputs=[audio_upload_in], outputs=[last_orig_state])
        audio_mic_in.change(fn=lambda p: p, inputs=[audio_mic_in], outputs=[last_orig_state])

        profile_dd.change(
            fn=lambda pid: gr.update(
                value=True,
                elem_classes=(
                    []
                    if DOMAIN_PROFILES.get(pid, DOMAIN_PROFILES["banking_call_center"]).allow_disable_rvox
                    else ["hidden-view"]
                ),
            ),
            inputs=[profile_dd],
            outputs=[embed_rvox_chk],
        )

        btn_deidentify.click(
            fn=run_shield_deidentify,
            inputs=[
                shield_src_mode,
                audio_upload_in,
                audio_mic_in,
                sample_dd,
                profile_dd,
                embed_rvox_chk,
                master_pw_in,
            ],
            outputs=[
                anon_audio_out,
                btn_download_anon,
                shield_kpi_html,
                whisper_asr_box,
                shield_mel_plot,
                last_anon_state,
            ],
        )

        # =====================================================================
        # VAULT SOURCE SWITCHER & TRANSFER EVENTS
        # =====================================================================
        def _set_vault_source(vmode: str):
            choices_upd = gr.update(choices=_list_recent_rvox_wavs()) if vmode == "recent" else gr.update()
            return (
                vmode,
                gr.update(elem_classes=["source-pane"] if vmode == "upload" else ["source-pane", "hidden-view"]),
                gr.update(elem_classes=["source-pane"] if vmode == "recent" else ["source-pane", "hidden-view"]),
                gr.update(variant="primary" if vmode == "upload" else "secondary"),
                gr.update(variant="primary" if vmode == "recent" else "secondary"),
                choices_upd,
            )

        vsrc_outputs = [
            vault_src_mode,
            vpane_upload,
            vpane_recent,
            btn_vmode_upload,
            btn_vmode_recent,
            recent_anon_dd,
        ]
        btn_vmode_upload.click(fn=lambda: _set_vault_source("upload"), inputs=[], outputs=vsrc_outputs)
        btn_vmode_recent.click(fn=lambda: _set_vault_source("recent"), inputs=[], outputs=vsrc_outputs)

        vault_wav_file.change(
            fn=_inspect_vault_file,
            inputs=[vault_wav_file],
            outputs=[vault_anon_preview, vault_status_html],
        )
        recent_anon_dd.change(
            fn=_inspect_vault_file,
            inputs=[recent_anon_dd],
            outputs=[vault_anon_preview, vault_status_html],
        )

        def _send_anon_to_vault(anon_path: str | None, orig_path: str | None, pw: str):
            if not anon_path or not Path(anon_path).exists():
                raise gr.Error("Hãy bấm 'Ẩn danh giọng nói' trước khi chuyển sang khôi phục!")
            prev_p, badge_html = _inspect_vault_file(anon_path)
            recent_choices = _list_recent_rvox_wavs()
            return (
                gr.update(elem_classes=["view-panel", "hidden-view"]),
                gr.update(elem_classes=["view-panel"]),
                gr.update(elem_classes=["studio-card", "hidden-view"]),
                gr.update(variant="secondary"),
                gr.update(variant="primary"),
                gr.update(variant="secondary"),
                "recent",
                gr.update(elem_classes=["source-pane", "hidden-view"]),
                gr.update(elem_classes=["source-pane"]),
                gr.update(variant="secondary"),
                gr.update(variant="primary"),
                gr.update(choices=recent_choices, value=anon_path),
                prev_p,
                badge_html,
                orig_path,
                pw,
            )

        btn_send_to_vault.click(
            fn=_send_anon_to_vault,
            inputs=[last_anon_state, last_orig_state, master_pw_in],
            outputs=[
                *nav_outputs,
                vault_src_mode,
                vpane_upload,
                vpane_recent,
                btn_vmode_upload,
                btn_vmode_recent,
                recent_anon_dd,
                vault_anon_preview,
                vault_status_html,
                vault_orig_optional,
                vault_master_pw,
            ],
        )

        btn_reidentify.click(
            fn=run_vault_reidentify,
            inputs=[vault_src_mode, vault_wav_file, recent_anon_dd, vault_master_pw, vault_orig_optional],
            outputs=[
                recon_audio_out,
                btn_download_recon,
                vault_kpi_html,
                vault_asr_box,
                vault_trio_plot,
            ],
        )

        btn_refresh_bench.click(
            fn=_load_benchmark_dashboard_data,
            inputs=[],
            outputs=[
                bench_kpi_html,
                bench_table,
                speakers_table,
                bench_dist_image,
                bench_bar_image,
                bench_trio_image,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ReVox Studio Web Interface.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    if args.preload:
        get_pipeline(load_asr=True)

    demo = build_revox_app()
    launch_kwargs: dict[str, Any] = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "allowed_paths": [str(OUTPUTS_DIR.resolve()), str(TEST_SAMPLES_DIR.resolve())],
    }
    launch_sig = inspect.signature(demo.launch).parameters
    if "css" in launch_sig and "css" not in inspect.signature(gr.Blocks.__init__).parameters:
        launch_kwargs["css"] = STUDIO_CSS

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
