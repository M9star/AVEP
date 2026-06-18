"""
Layer 3 — Metadata Bridge
Converts edit_plan.json → FCPXML + EDL timeline files.
Run:  python -m layer3_bridge.bridge
"""
import json
import opentimelineio as otio
from pathlib import Path
from config.settings import get_paths


def run(video_path: str, fps: float = 30.0):
    paths = get_paths(video_path)
    paths["inter_dir"].mkdir(parents=True, exist_ok=True)

    print("\n[L3] Loading edit plan...")
    with open(paths["edit_plan"]) as f:
        plan = json.load(f)

    timeline = otio.schema.Timeline(name="AVEP_Edit")
    track    = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

    for seg in plan.get("keep_segments", []):
        duration_frames = (seg["end"] - seg["start"]) * fps
        clip = otio.schema.Clip(
            name=seg.get("reason", "clip"),
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(seg["start"] * fps, fps),
                duration=otio.opentime.RationalTime(duration_frames, fps),
            )
        )
        track.append(clip)

    timeline.tracks.append(track)

    # Export FCPXML
    otio.adapters.write_to_file(timeline, str(paths["fcpxml_output"]))
    print(f"  [Bridge] FCPXML → {paths['fcpxml_output']}")

    # Export EDL
    otio.adapters.write_to_file(timeline, str(paths["edl_output"]))
    print(f"  [Bridge] EDL    → {paths['edl_output']}")

    print(f"\n[L3] ✓ Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AVEP Layer 3 — Metadata Bridge")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    run(args.video, args.fps)
