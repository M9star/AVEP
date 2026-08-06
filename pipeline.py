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
    parser.add_argument("--fps",          type=float, default=None,
                        help="Override source FPS (default: auto-detect)")
    parser.add_argument("--subject",      default="", help="Subject hint for transcript correction")
    parser.add_argument("--llm-provider", default=None,
                        help="Per-run provider: claude, openai, gemini, claude_code, or ollama:model")
    parser.add_argument("--agent",        action="store_true",
                        help="Run the persistent orchestrator + quality/revision loop")
    parser.add_argument("--goal",         default="Create a concise, natural rough cut.",
                        help="Editing goal used by agent mode and LLM planning")
    parser.add_argument("--max-attempts", type=int, default=3, choices=range(1, 6), metavar="1-5")
    parser.add_argument("--run-id",       default=None, help="Agent run ID (required with --resume)")
    parser.add_argument("--resume",       action="store_true", help="Resume a persisted agent run")
    parser.add_argument("--layer",        type=int, choices=[1,2,3,4],
                        help="Run only a specific layer (1-4)")
    args = parser.parse_args()

    if args.agent and args.layer:
        parser.error("--agent cannot be combined with --layer")
    if args.resume and not args.agent:
        parser.error("--resume requires --agent")
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")

    t0 = time.time()
    print("\n" + "="*60)
    print("  🎬 AVEP — Autonomous Video Editing Pipeline")
    print("="*60)

    if args.agent:
        from agents.orchestrator import AgentOrchestrator

        state = AgentOrchestrator().run(
            args.video,
            goal=args.goal,
            skip_denoise=args.skip_denoise,
            skip_llm=args.skip_llm,
            subject=args.subject,
            fps=args.fps,
            llm_provider=args.llm_provider,
            max_attempts=args.max_attempts,
            run_id=args.run_id,
            resume=args.resume,
        )
        print(f"\n  🤖 Agent run complete: {state.run_id} (attempt {state.attempt})")
        elapsed = time.time() - t0
        print(f"  ✅ Completed in {elapsed:.1f}s\n")
        return

    only = args.layer

    if not only or only == 1:
        print("\n▶ LAYER 1 — Perception")
        run_perception(args.video, args.skip_denoise)

    if not only or only == 2:
        print("\n▶ LAYER 2 — Decision Agent")
        run_decision(
            args.video,
            args.skip_llm,
            args.subject,
            llm_provider=args.llm_provider,
            editing_goal=args.goal,
        )

    if not only or only == 3:
        print("\n▶ LAYER 3 — Metadata Bridge")
        run_bridge(args.video, args.fps)

    if not only or only == 4:
        print("\n▶ LAYER 4 — Render")
        run_render(args.video)

    elapsed = time.time() - t0
    print("\n" + "="*60)
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
