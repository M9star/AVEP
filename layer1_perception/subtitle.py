"""
Layer 1 — SRT Subtitle Generator
Groups words into chunked phrases and writes .srt for preview.
"""
from pathlib import Path


def _format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_chunks(words: list[dict], max_words: int = 5, pause_gap: float = 0.6) -> list[dict]:
    """Group words into phrases by pause length or word count."""
    chunks, current = [], []
    for w in words:
        if current and (
            len(current) >= max_words
            or w["start"] - current[-1]["end"] > pause_gap
            or current[-1]["word"].endswith((".", "?", "!"))
        ):
            chunks.append({
                "text": " ".join(x["word"] for x in current),
                "start": current[0]["start"],
                "end": current[-1]["end"],
            })
            current = []
        current.append(w)
    if current:
        chunks.append({
            "text": " ".join(x["word"] for x in current),
            "start": current[0]["start"],
            "end": current[-1]["end"],
        })
    return chunks


def generate_srt(words: list[dict], output_path: str, max_words: int = 5, pause_gap: float = 0.6) -> str:
    """Generate .srt subtitle file from word-level timestamps."""
    chunks = words_to_chunks(words, max_words, pause_gap)
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(str(i))
        lines.append(f"{_format_ts(chunk['start'])} --> {_format_ts(chunk['end'])}")
        lines.append(chunk["text"])
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"  [Subtitle] Generated {len(chunks)} chunks → {output_path}")
    return output_path
