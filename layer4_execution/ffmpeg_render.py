"""
Layer 4 — FFmpeg Render (fallback, no DaVinci needed)
Applies the edit plan directly via FFmpeg filter_complex.
"""
import json
import subprocess
from pathlib import Path
from config.settings import get_paths


def detect_hw_encoder():
    """Pick best available H264 encoder."""
    import platform
    if platform.system() == "Darwin":
        return "h264_videotoolbox"
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            return "h264_nvenc"
    except FileNotFoundError:
        pass
    return "libx264"


def run(source_video: str):
    paths = get_paths(source_video)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    with open(paths["edit_plan"]) as f:
        plan = json.load(f)

    segments = plan.get("keep_segments", [])
    if not segments:
        print("[L4] No keep segments found in edit plan. Exiting.")
        return

    encoder = detect_hw_encoder()
    print(f"\n[L4] Rendering with encoder: {encoder}")

    # Build select filter
    select_parts  = "+".join(f"between(t,{s['start']},{s['end']})" for s in segments)
    filter_chain  = f"select='{select_parts}',setpts=N/FRAME_RATE/TB"
    afilter_chain = f"aselect='{select_parts}',asetpts=N/SR/TB"

    cmd = [
        "ffmpeg", "-y",
        "-i", source_video,
        "-vf", filter_chain,
        "-af", afilter_chain,
        "-c:v", encoder,
        "-b:v", "8000k",
        "-c:a", "aac",
        "-b:a", "192k",
        str(paths["final_output"]),
    ]

    print(f"  [L4] Running ffmpeg...")
    subprocess.run(cmd, check=True)
    print(f"\n[L4] ✓ Final cut → {paths['final_output']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AVEP Layer 4 — FFmpeg Render")
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    run(args.video)
