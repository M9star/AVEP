"""
Layer 3 — Metadata Bridge
Converts edit_plan.json → FCPXML + EDL timeline files.
Run:  python -m layer3_bridge.bridge
"""
import json
import opentimelineio as otio
from pathlib import Path
from config.settings import EDIT_PLAN, FCPXML_OUTPUT, EDL_OUTPUT, INTER_DIR


def run(fps: float = 30.0):
    INTER_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[L3] Loading edit plan...")
    with open(EDIT_PLAN) as f:
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
    otio.adapters.write_to_file(timeline, str(FCPXML_OUTPUT))
    print(f"  [Bridge] FCPXML → {FCPXML_OUTPUT}")

    # Export EDL
    otio.adapters.write_to_file(timeline, str(EDL_OUTPUT))
    print(f"  [Bridge] EDL    → {EDL_OUTPUT}")

    print(f"\n[L3] ✓ Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AVEP Layer 3 — Metadata Bridge")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    run(args.fps)
