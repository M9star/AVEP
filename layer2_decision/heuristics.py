"""
Layer 2 — Pre-LLM heuristics.
Fast local processing to reduce the LLM's workload and cost.
"""
import re
from config.settings import FILLER_WORDS, SILENCE_THRESHOLD_SEC


def flag_filler_words(words: list[dict]) -> list[dict]:
    """Returns list of word dicts that are filler words."""
    pattern = re.compile(
        r"^\b(" + "|".join(FILLER_WORDS) + r")\b$", re.IGNORECASE
    )
    return [w for w in words if pattern.match(w["word"].strip(".,!?"))]


def flag_long_silences(silences: list[dict]) -> list[dict]:
    """Returns silence segments longer than threshold."""
    return [s for s in silences if (s["end"] - s["start"]) >= SILENCE_THRESHOLD_SEC]


def build_sentences(words: list[dict]) -> list[dict]:
    """Groups words into rough sentences by punctuation or long pauses."""
    sentences, current = [], []
    for w in words:
        current.append(w)
        if w["word"].endswith((".", "?", "!")) or \
           (len(current) > 1 and w["start"] - current[-2]["end"] > 0.6):
            sentences.append({
                "text": " ".join(x["word"] for x in current),
                "start": current[0]["start"],
                "end": current[-1]["end"],
            })
            current = []
    if current:
        sentences.append({
            "text": " ".join(x["word"] for x in current),
            "start": current[0]["start"],
            "end": current[-1]["end"],
        })
    return sentences
