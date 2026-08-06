"""Editing agent: creates and revises validated edit plans."""
from layer2_decision.decision import run as run_decision
from layer2_decision.schemas import validate_edit_plan


class EditDecisionAgent:
    def create_plan(
        self,
        video_path: str,
        goal: str,
        skip_llm: bool,
        subject: str,
        llm_provider: str | None,
    ) -> dict:
        return run_decision(
            video_path,
            skip_llm,
            subject,
            llm_provider=llm_provider,
            editing_goal=goal,
        )

    def revise_plan(self, plan: dict, duration: float, min_segment_duration: float) -> dict:
        """Repair short/fragmented cuts, then rebuild the remove complement."""
        keep = [dict(segment) for segment in plan["keep_segments"]]
        keep.sort(key=lambda segment: segment["start"])

        merged = []
        for segment in keep:
            if merged and segment["start"] - merged[-1]["end"] <= min_segment_duration:
                merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
                merged[-1]["reason"] = "agent revision: merged nearby speech"
            else:
                merged.append(segment)

        padded = []
        padding = min_segment_duration / 2
        for segment in merged:
            if segment["end"] - segment["start"] < min_segment_duration:
                segment["start"] = max(0.0, segment["start"] - padding)
                segment["end"] = min(duration, segment["end"] + padding)
                segment["reason"] = "agent revision: padded short cut"
            if padded and segment["start"] < padded[-1]["end"]:
                padded[-1]["end"] = max(padded[-1]["end"], segment["end"])
                padded[-1]["reason"] = "agent revision: merged overlapping padding"
            else:
                padded.append(segment)

        remove = []
        cursor = 0.0
        for segment in padded:
            if segment["start"] > cursor:
                remove.append({
                    "start": round(cursor, 3),
                    "end": round(segment["start"], 3),
                    "reason": "agent revision: omitted interval",
                })
            cursor = max(cursor, segment["end"])
        if cursor < duration:
            remove.append({
                "start": round(cursor, 3),
                "end": round(duration, 3),
                "reason": "agent revision: omitted interval",
            })

        revised = {
            "keep_segments": padded,
            "remove_segments": remove,
            "flag_zoom": plan.get("flag_zoom", []),
        }
        return validate_edit_plan(revised, duration=duration)
