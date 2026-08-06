"""Typed contracts shared by AVEP agents and the orchestrator."""
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AgentRunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStep(str, Enum):
    PERCEPTION = "perception"
    EDIT_DECISION = "edit_decision"
    TIMELINE_EXPORT = "timeline_export"
    RENDER = "render"
    QUALITY_CONTROL = "quality_control"
    COMPLETE = "complete"


class QualitySeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class QualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: QualitySeverity
    action: str
    details: dict = Field(default_factory=dict)


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    attempt: int
    issues: list[QualityIssue] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)


class AgentRunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    video_path: str
    goal: str
    status: AgentRunStatus = AgentRunStatus.CREATED
    current_step: AgentStep = AgentStep.PERCEPTION
    attempt: int = 0
    max_attempts: int = Field(default=3, ge=1, le=5)
    completed_steps: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    quality_reports: list[QualityReport] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self):
        self.updated_at = datetime.now(timezone.utc)
