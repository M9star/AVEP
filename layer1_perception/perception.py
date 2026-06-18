"""
Layer 1 — Main Runner
Run:  python -m layer1_perception.perception --video data/input/myvideo.mp4
"""
import json
import argparse
from pathlib import Path
from layer1_perception.transcribe import transcribe
from layer1_perception.vad import detect_silence
from layer1_perception.denoise import denoise
from layer1_perception.subtitle import generate_srt
from config.settings import PERCEPTION_OUTPUT, PREVIEW_SRT, INTER_DIR


def run(video_path: str, skip_denoise: bool = False):
    video_path = Path(video_path)
    INTER_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — extract audio
    raw_audio = INTER_DIR / "raw_audio.wav"
    cleaned_audio = INTER_DIR / "clean_audio.wav"
    print(f"\n[L1] Extracting audio from {video_path.name}...")
    import subprocess
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

    # Step 4 — VAD silence map
    print("[L1] Detecting silence...")
    silences = detect_silence(audio_for_processing)

    # Step 5 — generate SRT for preview
    print("[L1] Generating subtitle preview...")
    generate_srt(words, str(PREVIEW_SRT))

    # Step 6 — write output
    output = {
        "source_video": str(video_path),
        "audio_used": audio_for_processing,
        "words": words,
        "silences": silences,
    }
    with open(PERCEPTION_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[L1] ✓ Done → {PERCEPTION_OUTPUT}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVEP Layer 1 — Perception")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--skip-denoise", action="store_true")
    args = parser.parse_args()
    run(args.video, args.skip_denoise)
