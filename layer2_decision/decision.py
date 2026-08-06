"""
Layer 2 — Main Runner
Run:  python -m layer2_decision.decision --video data/input/myvideo.mp4
"""
import json
import argparse
from config.settings import get_paths, probe_media, LLM_PROVIDER
from layer2_decision.heuristics import flag_filler_words, flag_long_silences, build_edit_plan
from layer2_decision.agent import call_llm
from layer2_decision.corrector import correct_transcript
from layer2_decision.schemas import validate_edit_plan
from layer1_perception.subtitle import generate_srt


def run(
    video_path: str,
    skip_llm: bool = False,
    subject_hint: str = "",
    llm_provider: str | None = None,
    editing_goal: str = "",
):
    # Resolve provider per invocation so one queued job cannot affect the next.
    provider = llm_provider or LLM_PROVIDER
    ollama_model = None
    if provider.startswith("ollama:"):
        ollama_model = provider.split(":", 1)[1]
        provider = "ollama"
        print(f"  [L2] Using Ollama model: {ollama_model}")
    print(f"  [L2] Using LLM provider: {provider}")

    paths = get_paths(video_path)
    paths["inter_dir"].mkdir(parents=True, exist_ok=True)

    print("\n[L2] Loading perception output...")
    with open(paths["raw_words"]) as f:
        raw_words_data = json.load(f)
    with open(paths["silence_noise_map"]) as f:
        silence_data = json.load(f)

    words = raw_words_data["words"]
    silences = silence_data["silences"]

    fillers  = flag_filler_words(words)
    long_silences = flag_long_silences(silences)
    print(f"  [Heuristics] Flagged {len(fillers)} filler words, {len(long_silences)} long silences")

    if skip_llm:
        print("  [Agent] Skipping LLM — using heuristic edit plan")
        edit_plan = build_edit_plan(words, silences)
        print(f"  [Heuristics] Generated {len(edit_plan['keep_segments'])} keep, {len(edit_plan['remove_segments'])} remove segments")
        corrected = {"corrected_words": words, "corrections_summary": [], "total_corrections": 0}
    else:
        print("[L2] Running transcript correction...")
        corrected = correct_transcript(words, subject_hint, provider, ollama_model)

        perception = {"words": corrected["corrected_words"], "silences": silences}
        edit_plan = call_llm(perception, provider, ollama_model, editing_goal)

    media = probe_media(video_path)
    edit_plan = validate_edit_plan(edit_plan, duration=media["duration"])

    # Save corrected transcript
    corrected_output = {
        "source_video": str(video_path),
        "subject_hint": subject_hint,
        "corrected_words": corrected["corrected_words"],
        "corrections_summary": corrected["corrections_summary"],
        "total_corrections": corrected["total_corrections"],
    }
    with open(paths["corrected_transcript"], "w") as f:
        json.dump(corrected_output, f, indent=2)
    print(f"  [Output] Corrected transcript → {paths['corrected_transcript']}")

    # Generate corrected SRT
    generate_srt(corrected["corrected_words"], str(paths["corrected_srt"]))
    print(f"  [Output] Corrected SRT → {paths['corrected_srt']}")

    # Save edit plan
    with open(paths["edit_plan"], "w") as f:
        json.dump(edit_plan, f, indent=2)

    print(f"\n[L2] ✓ Done → {paths['edit_plan']}")
    return edit_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVEP Layer 2 — Decision Agent")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--skip-llm", action="store_true", help="Run heuristics only, skip LLM call")
    parser.add_argument("--subject", default="", help="Subject hint for transcript correction (e.g. 'College Physics II')")
    args = parser.parse_args()
    run(args.video, args.skip_llm, args.subject)
