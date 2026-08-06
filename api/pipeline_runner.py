"""
Background pipeline runner with FIFO job queue.
One pipeline runs at a time. Uploads queue up and process sequentially.
"""
import asyncio
import traceback
from datetime import datetime, timezone
from api.models import LayerStatus, PipelineStatus, LayerInfo


LAYER_NAMES = {
    1: ("layer1_perception", "Perception — Speech to Text"),
    2: ("layer2_decision", "Decision — Edit Planning"),
    3: ("layer3_bridge", "Metadata Bridge — Timeline Export"),
    4: ("layer4_render", "Render — Final Cut"),
}


def _make_layers() -> dict[str, LayerInfo]:
    return {
        key: LayerInfo(name=desc)
        for key, (_, desc) in LAYER_NAMES.items()
    }


class JobQueue:
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._order: list[str] = []
        self._current_job_id: str | None = None
        self._worker_task: asyncio.Task | None = None

    def start_worker(self, jobs: dict, event_queues: dict):
        self._jobs = jobs
        self._event_queues = event_queues
        self._worker_task = asyncio.create_task(self._worker())

    async def enqueue(self, job_id: str):
        self._order.append(job_id)
        await self._queue.put(job_id)

    @property
    def current_job_id(self) -> str | None:
        return self._current_job_id

    def queue_position(self, job_id: str) -> int | None:
        if job_id not in self._order:
            return None
        return self._order.index(job_id) + 1

    def pending_count(self) -> int:
        return self._queue.qsize()

    def pending_jobs(self) -> list[str]:
        return list(self._order)

    async def _worker(self):
        while True:
            job_id = await self._queue.get()
            self._current_job_id = job_id

            job = self._jobs.get(job_id)
            if not job:
                self._order.remove(job_id)
                self._current_job_id = None
                continue

            await _run_pipeline(job, self._jobs, self._event_queues)

            if job_id in self._order:
                self._order.remove(job_id)
            self._current_job_id = None


job_queue = JobQueue()


async def _run_pipeline(job: dict, jobs: dict, event_queues: dict):
    if job.get("agent_mode", False):
        await _run_agent_pipeline(job, event_queues)
        return

    job_id = job["job_id"]
    video_path = job["video_path"]
    skip_denoise = job.get("skip_denoise", False)
    skip_llm = job.get("skip_llm", False)
    subject = job.get("subject", "")
    fps = job.get("fps")  # None → Layer 3 auto-detects from the video
    llm_provider = job.get("llm_provider")  # None → use .env default

    job["status"] = PipelineStatus.RUNNING
    _notify(event_queues, job_id, "pipeline_started", {"job_id": job_id})

    loop = asyncio.get_running_loop()

    def make_progress_cb(layer_num: int):
        # Called from a worker thread — schedule the SSE notify on the loop safely.
        def cb(message: str, pct=None):
            job.setdefault("progress", {})[layer_num] = {"message": message, "pct": pct}
            loop.call_soon_threadsafe(
                _notify, event_queues, job_id, "layer_progress",
                {"layer": layer_num, "message": message, "pct": pct},
            )
        return cb

    layer_funcs = {
        1: lambda: _run_layer1(video_path, skip_denoise, make_progress_cb(1)),
        2: lambda: _run_layer2(video_path, skip_llm, subject, llm_provider),
        3: lambda: _run_layer3(video_path, fps),
        4: lambda: _run_layer4(video_path),
    }

    for layer_num in range(1, 5):
        layer = job["layers"][layer_num]
        layer.status = LayerStatus.RUNNING
        layer.started_at = datetime.now(timezone.utc)
        _notify(event_queues, job_id, "layer_started", {
            "layer": layer_num,
            "name": layer.name,
        })

        try:
            await asyncio.to_thread(layer_funcs[layer_num])
            layer.status = LayerStatus.COMPLETED
            layer.completed_at = datetime.now(timezone.utc)
            _notify(event_queues, job_id, "layer_completed", {
                "layer": layer_num,
                "name": layer.name,
            })
        except Exception as e:
            layer.status = LayerStatus.FAILED
            layer.completed_at = datetime.now(timezone.utc)
            layer.error = str(e)
            job["status"] = PipelineStatus.FAILED
            job["error"] = f"Layer {layer_num} failed: {e}"
            _notify(event_queues, job_id, "layer_failed", {
                "layer": layer_num,
                "name": layer.name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            _notify(event_queues, job_id, "pipeline_failed", {
                "job_id": job_id,
                "failed_at_layer": layer_num,
            })
            return

    job["status"] = PipelineStatus.COMPLETED
    from config.settings import get_paths
    paths = get_paths(video_path)
    job["result"] = {
        "final_output": str(paths["final_output"]),
        "corrected_srt": str(paths["corrected_srt"]),
        "edit_plan": str(paths["edit_plan"]),
    }
    _notify(event_queues, job_id, "pipeline_completed", {
        "job_id": job_id,
        "result": job["result"],
    })


async def _run_agent_pipeline(job: dict, event_queues: dict):
    from agents.orchestrator import AgentOrchestrator
    from config.settings import get_paths

    job_id = job["job_id"]
    run_id = job.get("agent_run_id") or f"agent_{job_id}"
    job["agent_run_id"] = run_id
    job["status"] = PipelineStatus.RUNNING
    _notify(event_queues, job_id, "pipeline_started", {
        "job_id": job_id,
        "mode": "agent",
        "agent_run_id": run_id,
    })

    loop = asyncio.get_running_loop()

    def progress_cb(event_type: str, data: dict):
        loop.call_soon_threadsafe(
            _handle_agent_event,
            job,
            event_queues,
            event_type,
            data,
        )

    orchestrator = AgentOrchestrator(progress_cb=progress_cb)
    try:
        state = await asyncio.to_thread(
            orchestrator.run,
            job["video_path"],
            goal=job.get("editing_goal", "Create a concise, natural rough cut."),
            skip_denoise=job.get("skip_denoise", False),
            skip_llm=job.get("skip_llm", False),
            subject=job.get("subject", ""),
            fps=job.get("fps"),
            llm_provider=job.get("llm_provider"),
            max_attempts=job.get("max_attempts", 3),
            run_id=run_id,
        )
    except Exception as exc:
        job["status"] = PipelineStatus.FAILED
        job["error"] = f"Agent run failed: {exc}"
        failed_layer = next(
            (number for number, layer in job["layers"].items() if layer.status == LayerStatus.FAILED),
            None,
        )
        _notify(event_queues, job_id, "pipeline_failed", {
            "job_id": job_id,
            "failed_at_layer": failed_layer,
            "agent_run_id": run_id,
            "error": str(exc),
        })
        return

    paths = get_paths(job["video_path"])
    latest_report = state.quality_reports[-1] if state.quality_reports else None
    job["status"] = PipelineStatus.COMPLETED
    job["result"] = {
        "final_output": str(paths["final_output"]),
        "corrected_srt": str(paths["corrected_srt"]),
        "edit_plan": str(paths["edit_plan"]),
        "quality_report": str(paths["quality_report"]),
        "agent_run_id": state.run_id,
        "attempts": state.attempt,
        "quality": latest_report.model_dump(mode="json") if latest_report else None,
    }
    _notify(event_queues, job_id, "pipeline_completed", {
        "job_id": job_id,
        "mode": "agent",
        "agent_run_id": state.run_id,
        "result": job["result"],
    })


def _handle_agent_event(
    job: dict,
    event_queues: dict,
    event_type: str,
    data: dict,
):
    job_id = job["job_id"]
    layer_num = data.get("layer")
    if layer_num in job["layers"]:
        layer = job["layers"][layer_num]
        if event_type == "agent_tool_started":
            layer.status = LayerStatus.RUNNING
            layer.started_at = datetime.now(timezone.utc)
            layer.completed_at = None
            layer.error = None
            _notify(event_queues, job_id, "layer_started", {
                "layer": layer_num,
                "name": layer.name,
                "attempt": data.get("attempt"),
            })
        elif event_type == "agent_tool_completed":
            layer.status = LayerStatus.COMPLETED
            layer.completed_at = datetime.now(timezone.utc)
            _notify(event_queues, job_id, "layer_completed", {
                "layer": layer_num,
                "name": layer.name,
                "attempt": data.get("attempt"),
            })
        elif event_type == "agent_tool_failed":
            layer.status = LayerStatus.FAILED
            layer.completed_at = datetime.now(timezone.utc)
            layer.error = data.get("error")
            _notify(event_queues, job_id, "layer_failed", {
                "layer": layer_num,
                "name": layer.name,
                "error": layer.error,
                "attempt": data.get("attempt"),
            })
        elif event_type == "agent_tool_progress":
            job.setdefault("progress", {})[layer_num] = {
                "message": data.get("message"),
                "pct": data.get("pct"),
            }
            _notify(event_queues, job_id, "layer_progress", {
                "layer": layer_num,
                "message": data.get("message"),
                "pct": data.get("pct"),
            })

    _notify(event_queues, job_id, event_type, data)


def _run_layer1(video_path: str, skip_denoise: bool, progress_cb=None):
    from layer1_perception.perception import run as run_perception
    return run_perception(video_path, skip_denoise, progress_cb=progress_cb)


def _run_layer2(video_path: str, skip_llm: bool, subject: str, llm_provider: str | None):
    from layer2_decision.decision import run as run_decision
    return run_decision(video_path, skip_llm, subject, llm_provider=llm_provider)


def _run_layer3(video_path: str, fps: float):
    from layer3_bridge.bridge import run as run_bridge
    return run_bridge(video_path, fps)


def _run_layer4(video_path: str):
    from layer4_execution.ffmpeg_render import run as run_render
    return run_render(video_path)


def _notify(event_queues: dict, job_id: str, event_type: str, data: dict):
    queues = event_queues.get(job_id, [])
    for q in queues:
        q.put_nowait({"event": event_type, "data": data})
