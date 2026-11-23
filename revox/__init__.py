"""
ReVox: Reversible Neural Voice Anonymization & Biometric Identity Recovery System.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TARGET_POOL_DIR = DATA_DIR / "target_pool"
TEST_SAMPLES_DIR = DATA_DIR / "test_samples"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_CACHE_DIR = PROJECT_ROOT / "models_cache"

for _d in (DATA_DIR, TARGET_POOL_DIR, TEST_SAMPLES_DIR, OUTPUTS_DIR, MODELS_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(MODELS_CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODELS_CACHE_DIR / "torch_hub"))


def _apply_runtime_compat_patches() -> None:
    """Apply forward-compatibility shims for torchaudio, huggingface_hub, and SpeechBrain on Windows."""
    try:
        import torchaudio

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "get_audio_backend"):
            torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda _backend=None: None  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        import huggingface_hub
        import huggingface_hub.file_download as _hf_fd

        if os.name == "nt":
            _hf_fd.are_symlinks_supported = lambda *args, **kwargs: False  # type: ignore[assignment]

        sig = inspect.signature(huggingface_hub.hf_hub_download)
        if "use_auth_token" not in sig.parameters:
            _orig_download = huggingface_hub.hf_hub_download

            def _patched_hf_hub_download(*args, **kwargs):
                if "use_auth_token" in kwargs:
                    token_val = kwargs.pop("use_auth_token")
                    kwargs.setdefault("token", token_val)
                return _orig_download(*args, **kwargs)

            huggingface_hub.hf_hub_download = _patched_hf_hub_download
    except Exception:
        pass


_apply_runtime_compat_patches()

__version__ = "2.0.0"
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "TARGET_POOL_DIR",
    "TEST_SAMPLES_DIR",
    "OUTPUTS_DIR",
    "MODELS_CACHE_DIR",
    "__version__",
]
