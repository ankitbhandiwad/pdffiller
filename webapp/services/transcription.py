from __future__ import annotations

from webapp.exceptions import BadInputError, ExternalServiceError


def transcribe_audio_upload(integrations, audio_bytes: bytes, model_size: str) -> str:
    if not integrations.transcribe_available:
        raise ExternalServiceError("Transcription not available")
    if not audio_bytes:
        raise BadInputError("No audio provided")

    config = integrations.transcribe_config(model_size=model_size, device="cpu", compute_type="int8")
    return integrations.transcribe_audio_bytes(audio_bytes, config, suffix=".wav") or ""
