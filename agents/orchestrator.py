"""Goal-driven AVEP orchestrator with bounded quality/revision loops."""
import json
import uuid
from pathlib import Path
from typing import Callable

from agents.contracts import AgentRunState, AgentRunStatus, AgentStep, QualityReport
from agents.edit_agent import EditDecisionAgent
from agents.quality_agent import QualityControlAgent
from agents.state_store import AgentStateStore
from config.settings import AGENT_DB, get_paths, probe_media
from layer1_perception.perception import run as run_perception
from layer2_decision.schemas import validate_edit_plan
from layer3_bridge.bridge import run as run_bridge
from layer4_execution.ffmpeg_render import run as run_render


ProgressCallback = Callable[[str, dict], None]


class AgentOrchestrator:
    def __init__(
        self,
        store: AgentStateStore | None = None,
        edit_agent: EditDecisionAgent | None = None,
        quality_agent: QualityControlAgent | None = None,
        progress_cb: ProgressCallback | None = None,
    ):
        self.store = store or AgentStateStore(AGENT_DB)
        self.edit_agent = edit_agent or EditDecisionAgent()
        self.quality_agent = quality_agent or QualityControlAgent()
        self.progress_cb = progress_cb

    def _notify(self, state: AgentRunState, event_type: str, data: dict):
        payload = {"run_id": state.run_id, **data}
        self.store.record_event(state.run_id, event_type, payload)
        if self.progress_cb:
            self.progress_cb(event_type, payload)

    def _save(self, state: AgentRunState):
        self.store.save(state)

    def _run_tool(
        self,
        state: AgentRunState,
        step: AgentStep,
        layer: int | None,
        name: str,
        function: Callable,
    ):
        state.current_step = step
        self._save(state)
        self._notify(state, "agent_tool_started", {
            "step": step.value,
            "layer": layer,
            "name": name,
            "attempt": state.attempt,
        })
        try:
            result = function()
        except Exception as exc:
            self._notify(state, "agent_tool_failed", {
                "step": step.value,
                "layer": layer,
                "name": name,
                "attempt": state.attempt,
                "error": str(exc),
            })
            raise
        self._notify(state, "agent_tool_completed", {
            "step": step.value,
            "layer": layer,
            "name": name,
            "attempt": state.attempt,
        })
        return result

    def run(
        self,
        video_path: str,
        goal: str = "Create a concise, natural rough cut.",
        skip_denoise: bool = False,
        skip_llm: bool = False,
        subject: str = "",
        fps: float | None = None,
        llm_provider: str | None = None,
        max_attempts: int = 3,
        run_id: str | None = None,
        resume: bool = False,
    ) -> AgentRunState:
        video_path = str(Path(video_path).resolve())
        if resume:
            if not run_id:
                raise ValueError("run_id is required when resuming an agent run")
            state = self.store.load(run_id)
            if state is None:
                raise ValueError(f"Agent run not found: {run_id}")
            if Path(state.video_path).resolve() != Path(video_path):
                raise ValueError("Resume video does not match the stored agent run")
            if state.status == AgentRunStatus.COMPLETED:
                return state
        else:
            if run_id and self.store.load(run_id) is not None:
                raise ValueError(
                    f"Agent run already exists: {run_id}. Use resume=True to continue it."
                )
            state = AgentRunState(
                run_id=run_id or f"agent_{uuid.uuid4().hex[:12]}",
                video_path=video_path,
                goal=goal.strip() or "Create a concise, natural rough cut.",
                max_attempts=max_attempts,
            )
            self._save(state)

        paths = get_paths(video_path)
        state.status = AgentRunStatus.RUNNING
        state.error = None
        self._save(state)
        self._notify(state, "agent_started", {
            "goal": state.goal,
            "max_attempts": state.max_attempts,
            "resumed": resume,
        })

        try:
            media = probe_media(video_path)
            perception_ready = (
                "perception" in state.completed_steps
                and Path(paths["raw_words"]).is_file()
                and Path(paths["silence_noise_map"]).is_file()
            )
            if not perception_ready:
                self._run_tool(
                    state,
                    AgentStep.PERCEPTION,
                    1,
                    "Perception tool",
                    lambda: run_perception(
                        video_path,
                        skip_denoise,
                        progress_cb=lambda message, pct=None: self._notify(
                            state,
                            "agent_tool_progress",
                            {
                                "step": AgentStep.PERCEPTION.value,
                                "layer": 1,
                                "message": message,
                                "pct": pct,
                            },
                        ),
                    ),
                )
                state.completed_steps.append("perception")
                state.artifacts.update({
                    "raw_words": str(paths["raw_words"]),
                    "silence_noise_map": str(paths["silence_noise_map"]),
                })
                self._save(state)
            else:
                self._notify(state, "agent_tool_skipped", {
                    "step": AgentStep.PERCEPTION.value,
                    "layer": 1,
                    "name": "Perception tool",
                    "reason": "persisted artifacts are available",
                })

            plan_ready = (
                "edit_decision" in state.completed_steps
                and Path(paths["edit_plan"]).is_file()
            )
            if plan_ready:
                with open(paths["edit_plan"]) as file:
                    plan = validate_edit_plan(json.load(file), duration=media["duration"])
                self._notify(state, "agent_tool_skipped", {
                    "step": AgentStep.EDIT_DECISION.value,
                    "layer": 2,
                    "name": "Edit decision agent",
                    "reason": "persisted edit plan is available",
                })
            else:
                plan = self._run_tool(
                    state,
                    AgentStep.EDIT_DECISION,
                    2,
                    "Edit decision agent",
                    lambda: self.edit_agent.create_plan(
                        video_path,
                        state.goal,
                        skip_llm,
                        subject,
                        llm_provider,
                    ),
                )
                plan = validate_edit_plan(plan, duration=media["duration"])
                with open(paths["edit_plan"], "w") as file:
                    json.dump(plan, file, indent=2)
                state.completed_steps.append("edit_decision")
                state.artifacts["edit_plan"] = str(paths["edit_plan"])
                self._save(state)

            attempt = max(1, state.attempt or 1)
            while attempt <= state.max_attempts:
                state.attempt = attempt
                state.status = AgentRunStatus.RUNNING
                self._save(state)
                self._notify(state, "agent_attempt_started", {"attempt": attempt})

                self._run_tool(
                    state,
                    AgentStep.TIMELINE_EXPORT,
                    3,
                    "Timeline export tool",
                    lambda: run_bridge(video_path, fps),
                )
                state.artifacts.update({
                    "fcpxml": str(paths["fcpxml_output"]),
                    "edl": str(paths["edl_output"]),
                    "otio": str(paths["otio_output"]),
                })
                self._save(state)

                rendered_path = self._run_tool(
                    state,
                    AgentStep.RENDER,
                    4,
                    "Render tool",
                    lambda: run_render(video_path),
                )
                state.artifacts["final_output"] = rendered_path
                self._save(state)

                report: QualityReport = self._run_tool(
                    state,
                    AgentStep.QUALITY_CONTROL,
                    None,
                    "Quality-control agent",
                    lambda: self.quality_agent.review(
                        video_path,
                        rendered_path,
                        plan,
                        attempt,
                    ),
                )
                state.quality_reports.append(report)
                report_path = paths["inter_dir"] / f"quality_report_attempt_{attempt}.json"
                report_path.write_text(report.model_dump_json(indent=2))
                paths["quality_report"].write_text(report.model_dump_json(indent=2))
                state.artifacts["quality_report"] = str(paths["quality_report"])
                self._save(state)
                self._notify(state, "agent_quality_reviewed", report.model_dump(mode="json"))

                if report.approved:
                    state.status = AgentRunStatus.COMPLETED
                    state.current_step = AgentStep.COMPLETE
                    state.completed_steps.extend(["timeline_export", "render", "quality_control"])
                    self._save(state)
                    self._notify(state, "agent_completed", {
                        "attempt": attempt,
                        "artifacts": state.artifacts,
                    })
                    return state

                if attempt >= state.max_attempts:
                    issue_codes = ", ".join(issue.code for issue in report.issues)
                    raise RuntimeError(
                        f"Quality control rejected the render after {attempt} attempt(s): {issue_codes}"
                    )

                actions = {issue.action for issue in report.issues}
                state.status = AgentRunStatus.REVISING
                self._save(state)
                if "revise_plan" in actions:
                    plan = self.edit_agent.revise_plan(
                        plan,
                        media["duration"],
                        self.quality_agent.min_segment_duration,
                    )
                    with open(paths["edit_plan"], "w") as file:
                        json.dump(plan, file, indent=2)
                self._notify(state, "agent_revision_planned", {
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "actions": sorted(actions),
                })
                attempt += 1

            raise RuntimeError("Agent loop exited without an approved render")
        except Exception as exc:
            state.status = AgentRunStatus.FAILED
            state.error = str(exc)
            self._save(state)
            self._notify(state, "agent_failed", {
                "step": state.current_step.value,
                "attempt": state.attempt,
                "error": state.error,
            })
            raise
