"""
Layer 1 — Transcription
Uses Faster-Whisper to produce word-level timestamps.
"""
import json
from pathlib import Path
from faster_whisper import WhisperModel
from config.settings import WHISPER_MODEL, DEVICE


def get_device():
    if DEVICE != "AUTO":
        return DEVICE
    try:
        import torch
        if torch.cuda.is_available():    return "cuda"
        if torch.backends.mps.is_available(): return "mps"
    except ImportError:
        pass
    return "cpu"


def transcribe(audio_path: str) -> list[dict]:
    """
    Returns a list of word dicts:
    [{"word": "hello", "start": 0.0, "end": 0.4}, ...]
    """
    device = get_device()
    compute = "float16" if device == "cuda" else "int8"
    print(f"  [Whisper] Loading model={WHISPER_MODEL} device={device}")

    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
    )

    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    print(f"  [Whisper] Done — {len(words)} words, lang={info.language}")
    return words
