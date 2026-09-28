"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Lưu input + mốc thời gian bắt đầu, key theo request_id/user_id."""
        rid = request_id or user_id
        self._open[rid] = {
            "start": time.time(),
            "user_id": user_id,
            "text": text,
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Lưu output, lớp đã chặn (nếu có) và latency; append vào self.logs."""
        rid = request_id or user_id
        pending = self._open.pop(rid, None) or {}
        started = pending.get("start", time.time())
        latency_ms = round((time.time() - started) * 1000, 2)

        self.logs.append({
            "request_id": rid,
            "user_id": user_id,
            "input": pending.get("text", ""),
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
            "timestamp": utc_now_iso(),
        })

    def export_json(self, filepath: str | None = None):
        """Ghi toàn bộ log ra đĩa (JSON array), mặc định <repo>/outputs/audit_log.json."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
