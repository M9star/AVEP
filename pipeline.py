"""
AVEP — Master Pipeline Orchestrator
Runs all 4 layers end-to-end.

Usage:
  python pipeline.py --video data/input/myvideo.mp4
  python pipeline.py --video data/input/myvideo.mp4 --skip-denoise
  python pipeline.py --video data/input/myvideo.mp4 --skip-llm
  python pipeline.py --video data/input/myvideo.mp4 --layer 1   # run only layer 1
"""
import argparse
import time

from layer1_perception.perception import run as run_perception
from layer2_decision.decision     import run as run_decision
from layer3_bridge.bridge         import run as run_bridge
from layer4_execution.ffmpeg_render import run as run_render


def main():
    parser = argparse.ArgumentParser(description="AVEP — Full Pipeline")
    parser.add_argument("--video",        required=True, help="Path to input video")
    parser.add_argument("--skip-denoise", action="store_true")
    parser.add_argument("--skip-llm",     action="store_true")
    parser.add_argument("--fps",          type=float, default=30.0)
    parser.add_argument("--layer",        type=int, choices=[1,2,3,4],
                        help="Run only a specific layer (1-4)")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "="*60)
    print("  🎬 AVEP — Autonomous Video Editing Pipeline")
    print("="*60)

    only = args.layer

    if not only or only == 1:
        print("\n▶ LAYER 1 — Perception")
        run_perception(args.video, args.skip_denoise)

    if not only or only == 2:
        print("\n▶ LAYER 2 — Decision Agent")
        run_decision(args.skip_llm)

    if not only or only == 3:
        print("\n▶ LAYER 3 — Metadata Bridge")
        run_bridge(args.fps)

    if not only or only == 4:
        print("\n▶ LAYER 4 — Render")
        run_render(args.video)

    elapsed = time.time() - t0
    print("\n" + "="*60)
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
