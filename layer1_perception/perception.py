"""
Layer 1 — Main Runner
Run:  python -m layer1_perception.perception --video data/input/myvideo.mp4
"""
import json
import subprocess
import argparse
from pathlib import Path
from layer1_perception.transcribe import transcribe
from layer1_perception.vad import detect_silence, detect_speech_onsets
from layer1_perception.denoise import denoise
from layer1_perception.subtitle import generate_srt
from config.settings import get_paths


def _compute_sync_offset(words: list[dict], speech_onsets: list[float]) -> float:
    """Estimate timestamp drift by comparing Whisper word starts to VAD speech onsets.
    Returns median offset (positive = Whisper lags behind speech)."""
    if not words or not speech_onsets:
        return 0.0

    offsets = []
    onset_idx = 0
    for w in words:
        while onset_idx < len(speech_onsets) - 1 and speech_onsets[onset_idx + 1] <= w["start"]:
            onset_idx += 1
        if onset_idx < len(speech_onsets):
            diff = w["start"] - speech_onsets[onset_idx]
            if abs(diff) < 0.5:
                offsets.append(diff)

    if not offsets:
        return 0.0

    offsets.sort()
    return offsets[len(offsets) // 2]


def _apply_sync_offset(words: list[dict], offset: float) -> list[dict]:
    """Shift all word timestamps backward by offset to align with actual speech."""
    if abs(offset) < 0.01:
        return words
    return [
        {**w, "start": round(max(0, w["start"] - offset), 3),
               "end":   round(max(0, w["end"] - offset), 3)}
        for w in words
    ]


def run(video_path: str, skip_denoise: bool = False):
    video_path = Path(video_path)
    paths = get_paths(str(video_path))
    paths["inter_dir"].mkdir(parents=True, exist_ok=True)

    # Step 1 — extract audio
    raw_audio = paths["raw_audio"]
    cleaned_audio = paths["clean_audio"]
    print(f"\n[L1] Extracting audio from {video_path.name}...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", str(raw_audio)],
        check=True, capture_output=True
    )

    # Step 2 — optional denoise
    audio_for_processing = str(raw_audio)
    if not skip_denoise:
        print("[L1] Running noise reduction...")
        audio_for_processing = denoise(str(raw_audio), str(cleaned_audio))

    # Step 3 — transcribe
    print("[L1] Transcribing...")
    words = transcribe(audio_for_processing)

    # Step 4 — VAD silence map + speech onsets
    print("[L1] Detecting silence & speech onsets...")
    silences, speech_onsets = detect_silence(audio_for_processing, return_onsets=True)

    # Step 5 — fix timestamp sync drift
    offset = _compute_sync_offset(words, speech_onsets)
    if abs(offset) >= 0.01:
        print(f"  [Sync] Detected {offset*1000:.0f}ms drift — correcting timestamps")
        words = _apply_sync_offset(words, offset)

    # Step 6 — write split output files
    raw_words_data = {
        "source_video": str(video_path),
        "audio_used": audio_for_processing,
        "sync_offset_applied_ms": round(offset * 1000, 1),
        "words": words,
    }
    with open(paths["raw_words"], "w") as f:
        json.dump(raw_words_data, f, indent=2)
    print(f"  [Output] Word transcript → {paths['raw_words']}")

    silence_data = {
        "source_video": str(video_path),
        "silences": silences,
        "speech_onsets": speech_onsets,
    }
    with open(paths["silence_noise_map"], "w") as f:
        json.dump(silence_data, f, indent=2)
    print(f"  [Output] Silence/noise map → {paths['silence_noise_map']}")

    # Step 7 — generate SRT for preview
    print("[L1] Generating subtitle preview...")
    generate_srt(words, str(paths["preview_srt"]))

    # Step 8 — write combined output (backward compat)
    output = {
        "source_video": str(video_path),
        "audio_used": audio_for_processing,
        "words": words,
        "silences": silences,
    }
    with open(paths["perception_output"], "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[L1] ✓ Done → {paths['inter_dir']}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVEP Layer 1 — Perception")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--skip-denoise", action="store_true")
    args = parser.parse_args()
    run(args.video, args.skip_denoise)
