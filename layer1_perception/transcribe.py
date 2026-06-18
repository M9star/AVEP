"""
Layer 1 — Transcription
Uses Faster-Whisper to produce word-level timestamps.
"""
import json
from pathlib import Path
from faster_whisper import WhisperModel
from config.settings import WHISPER_MODEL, DEVICE


def detect_hardware() -> dict:
    """Detect all available compute devices. Returns full report."""
    report = {"cuda": False, "mps": False, "cpu": True, "gpu_name": None}
    try:
        import torch
        if torch.cuda.is_available():
            report["cuda"] = True
            report["gpu_name"] = torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            report["mps"] = True
    except ImportError:
        pass
    return report


def get_device() -> str:
    """
    Pick the best device for faster-whisper (CTranslate2).

    NOTE: CTranslate2 only supports 'cuda' and 'cpu' — NOT 'mps'.
    On Apple Silicon, MPS is detected (used elsewhere) but Whisper runs on CPU.
    """
    if DEVICE != "AUTO":
        return DEVICE

    hw = detect_hardware()
    if hw["cuda"]:
        return "cuda"
    # MPS unsupported by CTranslate2 → fall back to CPU on Apple Silicon
    return "cpu"


def transcribe(audio_path: str, progress_cb=None) -> list[dict]:
    """
    Returns a list of word dicts:
    [{"word": "hello", "start": 0.0, "end": 0.4}, ...]

    progress_cb(message: str, pct: float | None) — optional callback for live
    progress reporting (pct is 0-100 based on audio position).
    """
    def report(msg, pct=None):
        if progress_cb:
            progress_cb(msg, pct)

    hw = detect_hardware()
    detected = [k.upper() for k in ("cuda", "mps", "cpu") if hw[k]]
    device = get_device()
    compute = "float16" if device == "cuda" else "int8"
    gpu = f" ({hw['gpu_name']})" if hw["gpu_name"] else ""
    print(f"  [Whisper] Detected: {', '.join(detected)}{gpu} → using device={device}")
    report(f"Detected {', '.join(detected)} → Whisper on {device.upper()}{gpu}", None)
    report(f"Loading Whisper model ({WHISPER_MODEL}, {device})", None)

    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
    )

    total = info.duration or 0
    report(f"Transcribing {total:.0f}s of audio...", 0)

    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
        if total > 0:
            pct = min(100, round(seg.end / total * 100, 1))
            report(f"Transcribing... {pct:.0f}% ({len(words)} words)", pct)

    print(f"  [Whisper] Done — {len(words)} words, lang={info.language}")
    report(f"Transcribed {len(words)} words", 100)
    return words
