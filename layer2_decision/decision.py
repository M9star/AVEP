"""
Layer 2 — Main Runner
Run:  python -m layer2_decision.decision
"""
import json
import argparse
from config.settings import PERCEPTION_OUTPUT, EDIT_PLAN, INTER_DIR
from layer2_decision.heuristics import flag_filler_words, flag_long_silences
from layer2_decision.agent import call_llm


def run(skip_llm: bool = False):
    INTER_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[L2] Loading perception output...")
    with open(PERCEPTION_OUTPUT) as f:
        perception = json.load(f)

    # Run heuristics first (free & fast)
    fillers  = flag_filler_words(perception["words"])
    silences = flag_long_silences(perception["silences"])
    print(f"  [Heuristics] Flagged {len(fillers)} filler words, {len(silences)} long silences")

    if skip_llm:
        print("  [Agent] Skipping LLM (--skip-llm flag set)")
        edit_plan = {"keep_segments": [], "remove_segments": [], "flag_zoom": []}
    else:
        edit_plan = call_llm(perception)

    with open(EDIT_PLAN, "w") as f:
        json.dump(edit_plan, f, indent=2)

    print(f"\n[L2] ✓ Done → {EDIT_PLAN}")
    return edit_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AVEP Layer 2 — Decision Agent")
    parser.add_argument("--skip-llm", action="store_true", help="Run heuristics only, skip LLM call")
    args = parser.parse_args()
    run(args.skip_llm)
