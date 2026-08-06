"""Deterministic quality-control agent for rendered AVEP outputs."""
from pathlib import Path

from agents.contracts import QualityIssue, QualityReport, QualitySeverity
from config.settings import probe_media


class QualityControlAgent:
    def __init__(self, min_segment_duration: float = 0.35):
        self.min_segment_duration = min_segment_duration

    def review(self, source_video: str, rendered_video: str, plan: dict, attempt: int) -> QualityReport:
        issues = []
        source = probe_media(source_video)
        output_path = Path(rendered_video)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            issues.append(QualityIssue(
                code="missing_render",
                message="Render output is missing or empty.",
                severity=QualitySeverity.ERROR,
                action="rerender",
            ))
            return QualityReport(approved=False, attempt=attempt, issues=issues)

        rendered = probe_media(rendered_video)
        expected_duration = sum(
            segment["end"] - segment["start"] for segment in plan["keep_segments"]
        )
        duration_delta = abs(rendered["duration"] - expected_duration)
        tolerance = max(0.35, expected_duration * 0.08)
        if duration_delta > tolerance:
            issues.append(QualityIssue(
                code="duration_mismatch",
                message="Rendered duration does not match the edit plan.",
                severity=QualitySeverity.ERROR,
                action="rerender",
                details={"expected": expected_duration, "actual": rendered["duration"]},
            ))

        if source["has_audio"] and not rendered["has_audio"]:
            issues.append(QualityIssue(
                code="missing_audio",
                message="Source contains audio but the render does not.",
                severity=QualitySeverity.ERROR,
                action="rerender",
            ))

        short_segments = [
            segment for segment in plan["keep_segments"]
            if segment["end"] - segment["start"] < self.min_segment_duration
        ]
        if short_segments:
            issues.append(QualityIssue(
                code="short_keep_segments",
                message=f"{len(short_segments)} retained segment(s) are too short for a clean cut.",
                severity=QualitySeverity.WARNING,
                action="revise_plan",
                details={"segments": short_segments, "minimum": self.min_segment_duration},
            ))

        return QualityReport(
            approved=not issues,
            attempt=attempt,
            issues=issues,
            metrics={
                "source_duration": source["duration"],
                "expected_duration": round(expected_duration, 3),
                "rendered_duration": rendered["duration"],
                "duration_delta": round(duration_delta, 3),
                "keep_segments": len(plan["keep_segments"]),
                "output_size_bytes": output_path.stat().st_size,
                "has_audio": rendered["has_audio"],
            },
        )
