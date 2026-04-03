from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Optional

from faster_whisper import WhisperModel


@dataclass
class TranscribeConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    # Swap this to an API client later if you want hosted inference.


_MODEL: Optional[WhisperModel] = None
_MODEL_CFG: Optional[TranscribeConfig] = None


def get_model(cfg: TranscribeConfig) -> WhisperModel:
    global _MODEL, _MODEL_CFG
    if _MODEL is None or _MODEL_CFG != cfg:
        _MODEL = WhisperModel(cfg.model_size, device=cfg.device, compute_type=cfg.compute_type)
        _MODEL_CFG = cfg
    return _MODEL


def transcribe_audio_bytes(
    audio_bytes: bytes,
    cfg: Optional[TranscribeConfig] = None,
    suffix: str = ".wav",
) -> str:
    cfg = cfg or TranscribeConfig()
    model = get_model(cfg)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, _ = model.transcribe(
            tmp.name,
            language=None,
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
        )

    parts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()
