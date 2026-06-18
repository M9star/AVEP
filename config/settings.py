"""
Central configuration for AVEP.
All thresholds, model names, and paths live here.
"""
import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def detect_fps(video_path: str, default: float = 30.0) -> float:
    """Probe video frame rate via ffprobe. Falls back to default if detection fails."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        rate = json.loads(result.stdout)["streams"][0]["r_frame_rate"]  # e.g. "30000/1001"
        num, den = rate.split("/")
        fps = float(num) / float(den)
        return round(fps, 3) if fps > 0 else default
    except Exception:
        return default

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
INPUT_DIR   = DATA_DIR / "input"
INTER_DIR   = DATA_DIR / "intermediate"
OUTPUT_DIR  = DATA_DIR / "output"

PERCEPTION_OUTPUT = INTER_DIR / "perception_output.json"
PREVIEW_SRT       = INTER_DIR / "preview.srt"
EDIT_PLAN         = INTER_DIR / "edit_plan.json"
FCPXML_OUTPUT     = INTER_DIR / "timeline.fcpxml"
EDL_OUTPUT        = INTER_DIR / "timeline.edl"
FINAL_OUTPUT      = OUTPUT_DIR / "final_cut.mp4"


def get_paths(video_path: str) -> dict:
    """Return per-video output paths using video filename as subfolder."""
    stem = Path(video_path).stem
    inter = INTER_DIR / stem
    out   = OUTPUT_DIR / stem
    return {
        "inter_dir":             inter,
        "output_dir":            out,
        "raw_audio":             inter / "raw_audio.wav",
        "clean_audio":           inter / "clean_audio.wav",
        "perception_output":     inter / "perception_output.json",
        "raw_words":             inter / "raw_words.json",
        "silence_noise_map":     inter / "silence_noise_map.json",
        "corrected_transcript":  inter / "corrected_transcript.json",
        "preview_srt":           inter / "preview.srt",
        "corrected_srt":         inter / "corrected.srt",
        "edit_plan":             inter / "edit_plan.json",
        "fcpxml_output":         inter / "timeline.fcpxml",
        "edl_output":            inter / "timeline.edl",
        "final_output":          out   / "final_cut.mp4",
    }

# ── Models ───────────────────────────────────────────────────────────────────
WHISPER_MODEL  = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE         = os.getenv("DEVICE", "AUTO")   # AUTO = detect at runtime
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "claude")

# ── Decision Thresholds ──────────────────────────────────────────────────────
SILENCE_THRESHOLD_SEC = 0.8      # silences longer than this get removed
FILLER_WORDS          = ["um", "uh", "like", "you know", "basically", "literally"]
DUPLICATE_SIMILARITY  = 0.85     # Levenshtein ratio above this = duplicate take
