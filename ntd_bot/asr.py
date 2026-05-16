from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio_file(path: Path) -> str:
    model = _get_model()
    segments, _info = model.transcribe(str(path), language="ru")
    parts: list[str] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


def asr_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True
