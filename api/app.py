"""
AVEP — FastAPI Server
Upload video → run full pipeline → get notified on completion.

Start:  uvicorn api.app:app --reload
"""
import asyncio
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

from fastapi.staticfiles import StaticFiles

from config.settings import AGENT_DB, INPUT_DIR, DATA_DIR, INTER_DIR, OUTPUT_DIR, get_paths
from agents.state_store import AgentStateStore
from api.models import (
    PipelineStatus,
    JobResponse,
    UploadResponse,
    LayerInfo,
    LayerStatus,
)
from api.pipeline_runner import job_queue, _make_layers

app = FastAPI(title="AVEP — Autonomous Video Editing Pipeline", version="1.0.0")

# In-memory job store
jobs: dict[str, dict] = {}
# SSE event queues per job: {job_id: [asyncio.Queue, ...]}
event_queues: dict[str, list[asyncio.Queue]] = {}
agent_state_store = AgentStateStore(AGENT_DB)


@app.on_event("startup")
async def startup():
    job_queue.start_worker(jobs, event_queues)


ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


async def _enqueue_job(video_path: str, display_name: str,
                       skip_denoise: bool, skip_llm: bool,
                       subject: str, fps: float | None,
                       llm_provider: str | None = None,
                       agent_mode: bool = False,
                       editing_goal: str = "Create a concise, natural rough cut.",
                       max_attempts: int = 3) -> UploadResponse:
    """Register a job for an on-disk video and add it to the queue."""
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": PipelineStatus.QUEUED,
        "video_filename": display_name,
        "video_path": video_path,
        "created_at": datetime.now(timezone.utc),
        "layers": _make_layers(),
        "skip_denoise": skip_denoise,
        "skip_llm": skip_llm,
        "subject": subject,
        "fps": fps,
        "llm_provider": llm_provider,
        "agent_mode": agent_mode,
        "editing_goal": editing_goal,
        "max_attempts": max_attempts,
        "agent_run_id": None,
        "result": None,
        "error": None,
    }
    jobs[job_id] = job
    event_queues[job_id] = []

    await job_queue.enqueue(job_id)
    position = job_queue.queue_position(job_id)
    pending = job_queue.pending_count()

    return UploadResponse(
        job_id=job_id,
        message=f"Video queued. Position: {position}/{pending}.",
        status_url=f"/jobs/{job_id}",
        stream_url=f"/jobs/{job_id}/stream",
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    video: UploadFile = File(...),
    skip_denoise: bool = Form(False),
    skip_llm: bool = Form(False),
    subject: str = Form(""),
    fps: float | None = Form(None),
    llm_provider: str | None = Form(None),
    agent_mode: bool = Form(False),
    editing_goal: str = Form("Create a concise, natural rough cut."),
    max_attempts: int = Form(3),
):
    if not video.filename:
        raise HTTPException(400, "No filename provided")

    original_name = Path(video.filename).name
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {ALLOWED_EXT}")
    if not 1 <= max_attempts <= 5:
        raise HTTPException(400, "max_attempts must be between 1 and 5")

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    save_path = INPUT_DIR / save_name

    def _write():
        with open(save_path, "wb") as f:
            shutil.copyfileobj(video.file, f, length=1024 * 1024)
    await asyncio.to_thread(_write)

    return await _enqueue_job(str(save_path), original_name,
                              skip_denoise, skip_llm, subject, fps, llm_provider,
                              agent_mode, editing_goal, max_attempts)


@app.get("/input-files")
async def list_input_files():
    """List videos already present in data/input/ — pick these without uploading."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for item in sorted(INPUT_DIR.iterdir()):
        if item.is_file() and item.suffix.lower() in ALLOWED_EXT:
            files.append({
                "name": item.name,
                "size_bytes": item.stat().st_size,
                "size_mb": round(item.stat().st_size / 1024 / 1024, 1),
            })
    return {"files": files}


@app.post("/process", response_model=UploadResponse)
async def process_existing(
    filename: str = Form(...),
    skip_denoise: bool = Form(False),
    skip_llm: bool = Form(False),
    subject: str = Form(""),
    fps: float | None = Form(None),
    llm_provider: str | None = Form(None),
    agent_mode: bool = Form(False),
    editing_goal: str = Form("Create a concise, natural rough cut."),
    max_attempts: int = Form(3),
):
    """Queue a video that already lives in data/input/ — no upload needed."""
    safe_name = Path(filename).name
    video_path = (INPUT_DIR / safe_name).resolve()
    if not video_path.is_relative_to(INPUT_DIR.resolve()):
        raise HTTPException(403, "Access denied")
    if not video_path.is_file():
        raise HTTPException(404, f"File not found in data/input: {safe_name}")
    if video_path.suffix.lower() not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format: {video_path.suffix}")
    if not 1 <= max_attempts <= 5:
        raise HTTPException(400, "max_attempts must be between 1 and 5")

    return await _enqueue_job(str(video_path), safe_name,
                              skip_denoise, skip_llm, subject, fps, llm_provider,
                              agent_mode, editing_goal, max_attempts)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        video_filename=job["video_filename"],
        created_at=job["created_at"],
        layers=job["layers"],
        queue_position=job_queue.queue_position(job_id),
        result=job.get("result"),
        error=job.get("error"),
        agent_mode=job.get("agent_mode", False),
        agent_run_id=job.get("agent_run_id"),
    )


@app.get("/queue")
async def get_queue():
    """Show current queue state — what's processing and what's waiting."""
    current = job_queue.current_job_id
    pending = job_queue.pending_jobs()
    return {
        "processing": {
            "job_id": current,
            "video": jobs[current]["video_filename"] if current and current in jobs else None,
        } if current else None,
        "queued": [
            {
                "position": i + 1,
                "job_id": jid,
                "video": jobs[jid]["video_filename"] if jid in jobs else "unknown",
            }
            for i, jid in enumerate(pending)
            if jid != current
        ],
        "total_pending": len([j for j in pending if j != current]),
    }


@app.get("/jobs/{job_id}/stream")
async def stream_events(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    queue: asyncio.Queue = asyncio.Queue()
    event_queues.setdefault(job_id, []).append(queue)

    async def event_generator():
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=300)
                yield {
                    "event": msg["event"],
                    "data": json.dumps(msg["data"]),
                }
                if msg["event"] in ("pipeline_completed", "pipeline_failed"):
                    break
        except asyncio.TimeoutError:
            yield {"event": "timeout", "data": json.dumps({"message": "Stream timed out"})}
        finally:
            event_queues.get(job_id, []).remove(queue) if queue in event_queues.get(job_id, []) else None

    return EventSourceResponse(event_generator())


@app.get("/jobs")
async def list_jobs():
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "video_filename": j["video_filename"],
            "created_at": j["created_at"].isoformat(),
            "agent_mode": j.get("agent_mode", False),
            "agent_run_id": j.get("agent_run_id"),
        }
        for j in jobs.values()
    ]


@app.get("/agent-runs/{run_id}")
async def get_agent_run(run_id: str):
    """Return persisted orchestrator state and its complete decision/event history."""
    run = agent_state_store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Agent run {run_id} not found")
    return run


@app.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Fetch transcript, corrections, edit plan, and agent quality report."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    paths = get_paths(job["video_path"])
    result = {}

    # Raw transcript
    raw_words_path = Path(paths["raw_words"])
    if raw_words_path.exists():
        with open(raw_words_path) as f:
            data = json.load(f)
        result["transcript"] = {
            "word_count": len(data.get("words", [])),
            "words": data.get("words", []),
        }

    # Corrected transcript
    corrected_path = Path(paths["corrected_transcript"])
    if corrected_path.exists():
        with open(corrected_path) as f:
            data = json.load(f)
        result["corrections"] = {
            "total_corrections": data.get("total_corrections", 0),
            "corrections_summary": data.get("corrections_summary", []),
            "corrected_words": data.get("corrected_words", []),
        }

    # Edit plan
    edit_plan_path = Path(paths["edit_plan"])
    if edit_plan_path.exists():
        with open(edit_plan_path) as f:
            data = json.load(f)
        result["edit_plan"] = {
            "keep_count": len(data.get("keep_segments", [])),
            "remove_count": len(data.get("remove_segments", [])),
            "keep_segments": data.get("keep_segments", []),
            "remove_segments": data.get("remove_segments", []),
            "flag_zoom": data.get("flag_zoom", []),
        }

    # Latest agent quality-control decision (agent mode only)
    quality_report_path = Path(paths["quality_report"])
    if quality_report_path.exists():
        with open(quality_report_path) as f:
            result["quality_report"] = json.load(f)

    return result


@app.get("/ollama-models")
async def list_ollama_models():
    """Fetch available models from local Ollama instance."""
    import httpx
    ollama_url = "http://localhost:11434"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for m in data.get("models", []):
                size_gb = round(m.get("size", 0) / 1024**3, 1)
                models.append({"name": m["name"], "size": f"{size_gb}GB"})
            return {"models": models}
    except Exception:
        return {"models": []}


@app.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != PipelineStatus.COMPLETED:
        raise HTTPException(400, f"Job not completed yet. Status: {job['status']}")

    output_path = Path(job["result"]["final_output"])
    if not output_path.exists():
        raise HTTPException(404, "Output file not found on disk")
    return FileResponse(output_path, media_type="video/mp4", filename=f"avep_{job_id}.mp4")


@app.get("/jobs/{job_id}/files")
async def list_job_files(job_id: str):
    """List all intermediate + output files for a job."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    paths = get_paths(job["video_path"])
    files = {}
    for key, p in paths.items():
        if key in ("inter_dir", "output_dir"):
            continue
        p = Path(p)
        if p.exists():
            files[key] = {
                "path": str(p.relative_to(DATA_DIR)),
                "size_bytes": p.stat().st_size,
                "download_url": f"/data/{p.relative_to(DATA_DIR)}",
            }
    return {"job_id": job_id, "files": files}


@app.get("/data/{file_path:path}")
async def serve_data_file(file_path: str):
    """Serve any file from the data/ directory (intermediate, output, SRTs, JSON, etc.)."""
    full_path = DATA_DIR / file_path
    resolved = full_path.resolve()
    if not resolved.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(403, "Access denied")
    if not resolved.is_file():
        raise HTTPException(404, f"File not found: {file_path}")

    suffix = resolved.suffix.lower()
    media_types = {
        ".json": "application/json",
        ".srt": "text/plain",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".wav": "audio/wav",
        ".fcpxml": "application/xml",
        ".edl": "text/plain",
    }
    return FileResponse(resolved, media_type=media_types.get(suffix, "application/octet-stream"))


@app.get("/data")
async def browse_data(subdir: str = ""):
    """Browse the data/ directory tree. Use ?subdir=intermediate/myvideo to drill in."""
    target = DATA_DIR / subdir
    resolved = target.resolve()
    if not resolved.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(403, "Access denied")
    if not resolved.is_dir():
        raise HTTPException(404, f"Directory not found: {subdir}")

    entries = []
    for item in sorted(resolved.iterdir()):
        rel = item.relative_to(DATA_DIR)
        entry = {"name": item.name, "path": str(rel), "type": "dir" if item.is_dir() else "file"}
        if item.is_file():
            entry["size_bytes"] = item.stat().st_size
            entry["download_url"] = f"/data/{rel}"
        entries.append(entry)
    return {"directory": subdir or "/", "entries": entries}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "avep"}


# Mount static frontend last (catch-all for /)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
