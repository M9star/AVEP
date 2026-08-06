"""SQLite persistence for resumable AVEP agent runs and event history."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agents.contracts import AgentRunState


class AgentStateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_step TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_agent_events_run_id
                    ON agent_events(run_id, id);
                """
            )

    def save(self, state: AgentRunState):
        state.touch()
        payload = state.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, video_path, goal, status, current_step, attempt,
                    max_attempts, state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    video_path=excluded.video_path,
                    goal=excluded.goal,
                    status=excluded.status,
                    current_step=excluded.current_step,
                    attempt=excluded.attempt,
                    max_attempts=excluded.max_attempts,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    state.run_id,
                    state.video_path,
                    state.goal,
                    state.status.value,
                    state.current_step.value,
                    state.attempt,
                    state.max_attempts,
                    payload,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                ),
            )

    def load(self, run_id: str) -> AgentRunState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return AgentRunState.model_validate_json(row["state_json"]) if row else None

    def record_event(self, run_id: str, event_type: str, data: dict):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_events (run_id, event_type, data_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    event_type,
                    json.dumps(data, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_run(self, run_id: str) -> dict | None:
        state = self.load(run_id)
        if state is None:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_type, data_json, created_at
                FROM agent_events WHERE run_id = ? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        return {
            "state": state.model_dump(mode="json"),
            "events": [
                {
                    "event_type": row["event_type"],
                    "data": json.loads(row["data_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
        }
