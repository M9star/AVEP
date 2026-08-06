import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.contracts import (
    AgentRunState,
    AgentRunStatus,
    QualityIssue,
    QualityReport,
    QualitySeverity,
)
from agents.edit_agent import EditDecisionAgent
from agents.orchestrator import AgentOrchestrator
from agents.quality_agent import QualityControlAgent
from agents.state_store import AgentStateStore


class FakeEditAgent(EditDecisionAgent):
    def create_plan(self, video_path, goal, skip_llm, subject, llm_provider):
        return {
            "keep_segments": [{"start": 0.0, "end": 0.2, "reason": "short speech"}],
            "remove_segments": [{"start": 0.2, "end": 2.0, "reason": "silence"}],
            "flag_zoom": [],
        }


class FakeQualityAgent(QualityControlAgent):
    def __init__(self):
        super().__init__(min_segment_duration=0.35)
        self.calls = 0

    def review(self, source_video, rendered_video, plan, attempt):
        self.calls += 1
        if self.calls == 1:
            return QualityReport(
                approved=False,
                attempt=attempt,
                issues=[QualityIssue(
                    code="short_keep_segments",
                    message="Short segment",
                    severity=QualitySeverity.WARNING,
                    action="revise_plan",
                )],
            )
        return QualityReport(
            approved=True,
            attempt=attempt,
            metrics={"rendered_duration": 0.375},
        )


class AgentRuntimeTests(unittest.TestCase):
    def test_new_run_cannot_overwrite_persisted_run_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AgentStateStore(Path(temp_dir) / "agent.sqlite3")
            store.save(AgentRunState(
                run_id="agent_existing",
                video_path=str(Path(temp_dir) / "source.mp4"),
                goal="Existing goal",
            ))
            orchestrator = AgentOrchestrator(store=store)

            with self.assertRaisesRegex(ValueError, "already exists"):
                orchestrator.run(
                    str(Path(temp_dir) / "source.mp4"),
                    run_id="agent_existing",
                )

    def test_orchestrator_revises_then_persists_approved_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            intermediate = root / "intermediate"
            output = root / "output"
            intermediate.mkdir()
            output.mkdir()
            paths = {
                "inter_dir": intermediate,
                "raw_words": intermediate / "raw_words.json",
                "silence_noise_map": intermediate / "silence_noise_map.json",
                "edit_plan": intermediate / "edit_plan.json",
                "corrected_srt": intermediate / "corrected.srt",
                "fcpxml_output": intermediate / "timeline.fcpxml",
                "edl_output": intermediate / "timeline.edl",
                "otio_output": intermediate / "timeline.otio",
                "quality_report": intermediate / "quality_report.json",
                "final_output": output / "final_cut.mp4",
            }
            store = AgentStateStore(root / "agent.sqlite3")
            quality_agent = FakeQualityAgent()
            orchestrator = AgentOrchestrator(
                store=store,
                edit_agent=FakeEditAgent(),
                quality_agent=quality_agent,
            )

            def perception(*args, **kwargs):
                paths["raw_words"].write_text("{}")
                paths["silence_noise_map"].write_text("{}")

            def export(*args, **kwargs):
                for key in ("fcpxml_output", "edl_output", "otio_output"):
                    paths[key].write_text("artifact")

            def render(*args, **kwargs):
                paths["final_output"].write_bytes(b"rendered")
                return str(paths["final_output"])

            with (
                patch("agents.orchestrator.get_paths", return_value=paths),
                patch("agents.orchestrator.probe_media", return_value={
                    "duration": 2.0, "fps": 30.0, "has_audio": True, "has_video": True,
                }),
                patch("agents.orchestrator.run_perception", side_effect=perception),
                patch("agents.orchestrator.run_bridge", side_effect=export),
                patch("agents.orchestrator.run_render", side_effect=render),
            ):
                state = orchestrator.run(
                    str(source),
                    goal="Keep natural pacing",
                    skip_llm=True,
                    max_attempts=2,
                    run_id="agent_test",
                )

            self.assertEqual(state.status, AgentRunStatus.COMPLETED)
            self.assertEqual(state.attempt, 2)
            self.assertEqual(quality_agent.calls, 2)
            persisted = store.get_run("agent_test")
            self.assertEqual(persisted["state"]["status"], "completed")
            event_types = [event["event_type"] for event in persisted["events"]]
            self.assertIn("agent_revision_planned", event_types)
            self.assertIn("agent_completed", event_types)

    def test_quality_agent_rejects_short_segments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rendered = Path(temp_dir) / "rendered.mp4"
            rendered.write_bytes(b"video")
            media = {"duration": 1.0, "fps": 30.0, "has_audio": True, "has_video": True}
            plan = {
                "keep_segments": [{"start": 0.0, "end": 0.2, "reason": "short"}],
                "remove_segments": [{"start": 0.2, "end": 1.0, "reason": "cut"}],
                "flag_zoom": [],
            }
            with patch("agents.quality_agent.probe_media", return_value=media):
                report = QualityControlAgent().review("source.mp4", str(rendered), plan, 1)

            self.assertFalse(report.approved)
            self.assertEqual(report.issues[0].code, "duration_mismatch")
            self.assertIn("short_keep_segments", [issue.code for issue in report.issues])


if __name__ == "__main__":
    unittest.main()
