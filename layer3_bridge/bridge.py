"""
Layer 3 — Metadata Bridge
Converts edit_plan.json → FCPXML + EDL timeline files.
Run:  python -m layer3_bridge.bridge
"""
import json
import opentimelineio as otio
from pathlib import Path
from config.settings import get_paths, probe_media
from layer2_decision.schemas import validate_edit_plan


def _build_timeline(
    video_path: str,
    segments: list[dict],
    fps: float,
    duration: float,
    include_audio: bool,
) -> otio.schema.Timeline:
    timeline = otio.schema.Timeline(name="AVEP_Edit")
    source_url = Path(video_path).resolve().as_uri()
    available_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, fps),
        duration=otio.opentime.RationalTime(duration * fps, fps),
    )

    def make_clip(segment: dict) -> otio.schema.Clip:
        return otio.schema.Clip(
            name=Path(video_path).name,
            media_reference=otio.schema.ExternalReference(
                target_url=source_url,
                available_range=available_range,
            ),
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(segment["start"] * fps, fps),
                duration=otio.opentime.RationalTime(
                    (segment["end"] - segment["start"]) * fps, fps
                ),
            ),
            metadata={"avep": {"reason": segment["reason"]}},
        )

    video_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    video_track.extend(make_clip(segment) for segment in segments)
    timeline.tracks.append(video_track)

    if include_audio:
        audio_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
        audio_track.extend(make_clip(segment) for segment in segments)
        timeline.tracks.append(audio_track)

    return timeline


def run(video_path: str, fps: float | None = None):
    paths = get_paths(video_path)
    paths["inter_dir"].mkdir(parents=True, exist_ok=True)

    media = probe_media(video_path)
    if fps is None:
        fps = media["fps"]
        print(f"[L3] Auto-detected FPS: {fps}")
    if fps <= 0:
        raise ValueError("FPS must be greater than zero")

    print("\n[L3] Loading edit plan...")
    with open(paths["edit_plan"]) as f:
        plan = validate_edit_plan(json.load(f), duration=media["duration"])

    segments = plan["keep_segments"]
    timeline = _build_timeline(
        video_path,
        segments,
        fps,
        media["duration"],
        include_audio=media["has_audio"],
    )

    # Export FCPXML
    otio.adapters.write_to_file(timeline, str(paths["fcpxml_output"]))
    print(f"  [Bridge] FCPXML → {paths['fcpxml_output']}")

    # CMX 3600 is video-only; write a dedicated one-track timeline.
    edl_timeline = _build_timeline(
        video_path,
        segments,
        fps,
        media["duration"],
        include_audio=False,
    )
    otio.adapters.write_to_file(edl_timeline, str(paths["edl_output"]))
    print(f"  [Bridge] EDL    → {paths['edl_output']}")

    otio.adapters.write_to_file(timeline, str(paths["otio_output"]))
    print(f"  [Bridge] OTIO   → {paths['otio_output']}")

    print(f"\n[L3] ✓ Done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AVEP Layer 3 — Metadata Bridge")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()
    run(args.video, args.fps)
