"""
Layer 4 — FFmpeg Render (fallback, no DaVinci needed)
Applies the edit plan directly via FFmpeg filter_complex.
"""
import json
import subprocess
from pathlib import Path
from config.settings import get_paths, probe_media
from layer2_decision.schemas import validate_edit_plan


def detect_hw_encoder():
    """Pick best available H264 encoder."""
    import platform

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        )
        encoders = result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("ffmpeg is not installed or cannot list encoders") from exc

    if platform.system() == "Darwin" and "h264_videotoolbox" in encoders:
        return "h264_videotoolbox"
    if "h264_nvenc" in encoders:
        return "h264_nvenc"
    if "libx264" not in encoders:
        raise RuntimeError("No supported H.264 encoder found in ffmpeg")
    return "libx264"


def run(source_video: str):
    paths = get_paths(source_video)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    media = probe_media(source_video)

    with open(paths["edit_plan"]) as f:
        plan = validate_edit_plan(json.load(f), duration=media["duration"])

    segments = plan["keep_segments"]

    encoder = detect_hw_encoder()
    print(f"\n[L4] Rendering with encoder: {encoder}")

    # Build select filter
    select_parts  = "+".join(f"between(t,{s['start']},{s['end']})" for s in segments)
    filter_chain  = f"select='{select_parts}',setpts=N/FRAME_RATE/TB"
    afilter_chain = f"aselect='{select_parts}',asetpts=N/SR/TB"

    partial_output = paths["output_dir"] / "final_cut.partial.mp4"
    partial_output.unlink(missing_ok=True)

    def build_command(video_encoder: str) -> list[str]:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", source_video,
            "-vf", filter_chain,
            "-c:v", video_encoder,
            "-b:v", "8000k",
        ]
        if media["has_audio"]:
            command.extend(["-af", afilter_chain, "-c:a", "aac", "-b:a", "192k"])
        command.append(str(partial_output))
        return command

    print(f"  [L4] Running ffmpeg...")
    try:
        try:
            subprocess.run(build_command(encoder), check=True)
        except subprocess.CalledProcessError:
            if encoder == "libx264":
                raise
            partial_output.unlink(missing_ok=True)
            print(f"  [L4] {encoder} unavailable at runtime — retrying with libx264")
            subprocess.run(build_command("libx264"), check=True)
        rendered = probe_media(str(partial_output))
        if partial_output.stat().st_size == 0 or rendered["duration"] <= 0:
            raise RuntimeError("ffmpeg produced an invalid output file")
        partial_output.replace(paths["final_output"])
    except Exception:
        partial_output.unlink(missing_ok=True)
        raise
    print(f"\n[L4] ✓ Final cut → {paths['final_output']}")
    return str(paths["final_output"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AVEP Layer 4 — FFmpeg Render")
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    run(args.video)
