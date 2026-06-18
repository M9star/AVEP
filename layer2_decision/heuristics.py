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


def build_edit_plan(words: list[dict], silences: list[dict]) -> dict:
    """Build an edit plan from heuristics alone — no LLM needed.
    Removes long silences and filler words, keeps everything else."""
    if not words:
        return {"keep_segments": [], "remove_segments": [], "flag_zoom": []}

    fillers = set(id(w) for w in flag_filler_words(words))
    long_sil = flag_long_silences(silences)

    # Build remove set: long silences + filler word spans
    remove = []
    for s in long_sil:
        remove.append({"start": s["start"], "end": s["end"], "reason": "long silence"})
    for w in words:
        if id(w) in fillers:
            remove.append({"start": w["start"], "end": w["end"], "reason": f"filler: {w['word']}"})

    remove.sort(key=lambda x: x["start"])

    # Merge overlapping remove segments
    merged_remove = []
    for seg in remove:
        if merged_remove and seg["start"] <= merged_remove[-1]["end"] + 0.05:
            merged_remove[-1]["end"] = max(merged_remove[-1]["end"], seg["end"])
            merged_remove[-1]["reason"] += f"; {seg['reason']}"
        else:
            merged_remove.append(dict(seg))

    # Invert remove → keep (from first word to last word)
    video_start = words[0]["start"]
    video_end = words[-1]["end"]
    keep = []
    cursor = video_start

    for seg in merged_remove:
        if seg["start"] > cursor + 0.05:
            keep.append({"start": round(cursor, 3), "end": round(seg["start"], 3), "reason": "speech"})
        cursor = seg["end"]

    if cursor < video_end - 0.05:
        keep.append({"start": round(cursor, 3), "end": round(video_end, 3), "reason": "speech"})

    return {"keep_segments": keep, "remove_segments": merged_remove, "flag_zoom": []}


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
