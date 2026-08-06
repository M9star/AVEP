from pydantic import BaseModel
from enum import Enum
from datetime import datetime


class LayerStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LayerInfo(BaseModel):
    name: str
    status: LayerStatus = LayerStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class JobResponse(BaseModel):
    job_id: str
    status: PipelineStatus
    video_filename: str
    created_at: datetime
    layers: dict[int, LayerInfo]
    queue_position: int | None = None
    result: dict | None = None
    error: str | None = None
    agent_mode: bool = False
    agent_run_id: str | None = None


class UploadResponse(BaseModel):
    job_id: str
    message: str
    status_url: str
    stream_url: str
