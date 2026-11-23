"""
Cryptographic RIFF WAVE Container Engine for ReVox ('RVOX' custom chunk).
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

RVOX_CHUNK_ID = b"RVOX"
RVOX_PAYLOAD_MAGIC = b"RVX1"
RVOX_AAD = b"REVOX-V1-AUTHENTICATED-ACOUSTIC-VAULT"

SALT_BYTES = 16
NONCE_BYTES = 12
KEY_BYTES = 32
PBKDF2_ITERATIONS = 200_000


class ReVoxCryptoError(Exception):
    """Base exception for ReVox cryptographic or container errors."""


class ReVoxChunkMissingError(ReVoxCryptoError):
    """Raised when a WAV file does not contain an embedded 'RVOX' chunk."""


class ReVoxAuthError(ReVoxCryptoError):
    """Raised when Master Password verification fails (AES-256-GCM InvalidTag) or payload is corrupted."""


def derive_aes256_key(master_password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> bytes:
    """Derive a 256-bit cryptographic key from a UTF-8 Master Password using PBKDF2-HMAC-SHA256."""
    if not master_password or not master_password.strip():
        raise ValueError("Master Password không được để trống khi mã hóa/giải mã khóa RVOX.")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(master_password.encode("utf-8"))


def encrypt_payload_aes_gcm(plaintext: bytes, master_password: str) -> bytes:
    """Encrypt compressed codebook bytes with AES-256-GCM derived from Master Password."""
    salt = os.urandom(SALT_BYTES)
    nonce = os.urandom(NONCE_BYTES)
    key = derive_aes256_key(master_password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, RVOX_AAD)
    return RVOX_PAYLOAD_MAGIC + salt + nonce + ciphertext


def decrypt_payload_aes_gcm(encrypted_blob: bytes, master_password: str) -> bytes:
    """Authenticate and decrypt an 'RVOX' chunk payload using Master Password."""
    header_len = len(RVOX_PAYLOAD_MAGIC) + SALT_BYTES + NONCE_BYTES
    if len(encrypted_blob) <= header_len + 16:
        raise ReVoxAuthError("Dữ liệu chunk RVOX quá ngắn hoặc bị hỏng cấu trúc.")

    magic = encrypted_blob[:4]
    if magic != RVOX_PAYLOAD_MAGIC:
        raise ReVoxAuthError(f"Định dạng chunk RVOX không hợp lệ (magic={magic!r}).")

    salt = encrypted_blob[4 : 4 + SALT_BYTES]
    nonce = encrypted_blob[4 + SALT_BYTES : header_len]
    ciphertext = encrypted_blob[header_len:]

    key = derive_aes256_key(master_password, salt)
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ciphertext, RVOX_AAD)
    except InvalidTag as exc:
        raise ReVoxAuthError(
            "Sai mật khẩu Master Password hoặc dữ liệu khóa RVOX đã bị thay đổi!"
        ) from exc


def _iter_riff_chunks(raw_wav: bytes) -> list[tuple[bytes, bytes]]:
    if len(raw_wav) < 12 or raw_wav[:4] != b"RIFF" or raw_wav[8:12] != b"WAVE":
        raise ReVoxCryptoError("File đầu vào không phải định dạng chuẩn RIFF WAVE (.wav).")

    chunks: list[tuple[bytes, bytes]] = []
    offset = 12
    total_len = len(raw_wav)

    while offset + 8 <= total_len:
        chunk_id = raw_wav[offset : offset + 4]
        (chunk_size,) = struct.unpack("<I", raw_wav[offset + 4 : offset + 8])
        data_start = offset + 8
        data_end = data_start + chunk_size
        if data_end > total_len:
            chunk_payload = raw_wav[data_start:total_len]
            chunks.append((chunk_id, chunk_payload))
            break
        chunk_payload = raw_wav[data_start:data_end]
        chunks.append((chunk_id, chunk_payload))
        offset = data_end + (chunk_size % 2)

    return chunks


def _assemble_riff_wave(chunks: list[tuple[bytes, bytes]]) -> bytes:
    body_parts: list[bytes] = [b"WAVE"]
    for chunk_id, payload in chunks:
        if len(chunk_id) != 4:
            raise ValueError(f"RIFF chunk_id must be 4 bytes, got {chunk_id!r}")
        size = len(payload)
        body_parts.append(chunk_id)
        body_parts.append(struct.pack("<I", size))
        body_parts.append(payload)
        if size % 2 == 1:
            body_parts.append(b"\x00")

    wave_body = b"".join(body_parts)
    riff_header = b"RIFF" + struct.pack("<I", len(wave_body))
    return riff_header + wave_body


def embed_rvox_chunk(
    wav_path: str | Path,
    encrypted_payload: bytes,
    output_path: str | Path | None = None,
) -> Path:
    src = Path(wav_path)
    dst = Path(output_path) if output_path is not None else src
    raw_wav = src.read_bytes()
    chunks = _iter_riff_chunks(raw_wav)
    filtered_chunks = [(cid, cdata) for cid, cdata in chunks if cid != RVOX_CHUNK_ID]
    filtered_chunks.append((RVOX_CHUNK_ID, encrypted_payload))
    assembled = _assemble_riff_wave(filtered_chunks)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(assembled)
    return dst


def extract_rvox_chunk(wav_path: str | Path) -> bytes | None:
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {p}")

    with p.open("rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ReVoxCryptoError(f"File '{p.name}' không đúng chuẩn RIFF WAVE (.wav).")

        file_size = p.stat().st_size
        offset = 12
        while offset + 8 <= file_size:
            f.seek(offset)
            chunk_hdr = f.read(8)
            if len(chunk_hdr) < 8:
                break
            chunk_id = chunk_hdr[:4]
            (chunk_size,) = struct.unpack("<I", chunk_hdr[4:8])
            if chunk_id == RVOX_CHUNK_ID:
                return f.read(chunk_size)
            offset += 8 + chunk_size + (chunk_size % 2)

    return None


def inspect_rvox_chunk(wav_path: str | Path) -> dict[str, Any]:
    payload = extract_rvox_chunk(wav_path)
    if payload is None:
        return {"has_rvox": False, "chunk_bytes": 0, "chunk_kb": 0.0, "valid_magic": False}
    return {
        "has_rvox": True,
        "chunk_bytes": len(payload),
        "chunk_kb": round(len(payload) / 1024.0, 2),
        "valid_magic": payload[:4] == RVOX_PAYLOAD_MAGIC,
    }


def save_wav_with_optional_rvox(
    output_path: str | Path,
    waveform: np.ndarray,
    sample_rate: int = 16000,
    plaintext_rvox_payload: bytes | None = None,
    master_password: str | None = None,
) -> dict[str, Any]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    wav_np = np.asarray(waveform, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(wav_np))) if wav_np.size > 0 else 0.0
    if peak > 0.99:
        wav_np = wav_np * (0.98 / peak)

    sf.write(str(out), wav_np, sample_rate, subtype="PCM_16", format="WAV")

    if plaintext_rvox_payload is not None:
        if not master_password:
            raise ValueError("Cần cung cấp Master Password để mã hóa chunk RVOX.")
        encrypted = encrypt_payload_aes_gcm(plaintext_rvox_payload, master_password)
        embed_rvox_chunk(out, encrypted, out)
        return {
            "output_path": str(out),
            "has_rvox": True,
            "chunk_bytes": len(encrypted),
            "chunk_kb": round(len(encrypted) / 1024.0, 2),
        }

    return {
        "output_path": str(out),
        "has_rvox": False,
        "chunk_bytes": 0,
        "chunk_kb": 0.0,
    }
