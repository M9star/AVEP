"""
Central configuration for AVEP.
All thresholds, model names, and paths live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
INPUT_DIR   = DATA_DIR / "input"
INTER_DIR   = DATA_DIR / "intermediate"
OUTPUT_DIR  = DATA_DIR / "output"

PERCEPTION_OUTPUT = INTER_DIR / "perception_output.json"
EDIT_PLAN         = INTER_DIR / "edit_plan.json"
FCPXML_OUTPUT     = INTER_DIR / "timeline.fcpxml"
EDL_OUTPUT        = INTER_DIR / "timeline.edl"
FINAL_OUTPUT      = OUTPUT_DIR / "final_cut.mp4"

# ── Models ───────────────────────────────────────────────────────────────────
WHISPER_MODEL  = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE         = os.getenv("DEVICE", "AUTO")   # AUTO = detect at runtime
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "openai")

# ── Decision Thresholds ──────────────────────────────────────────────────────
SILENCE_THRESHOLD_SEC = 0.8      # silences longer than this get removed
FILLER_WORDS          = ["um", "uh", "like", "you know", "basically", "literally"]
DUPLICATE_SIMILARITY  = 0.85     # Levenshtein ratio above this = duplicate take
