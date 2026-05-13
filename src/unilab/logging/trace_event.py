from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _to_us(timestamp_ns: int) -> float:
    return timestamp_ns / 1000.0


class TraceRecorder:
    """Minimal Chrome trace / Perfetto event recorder."""

    _GPU_TID = 9001

    def __init__(self, process_name: str, *, pid: int | None = None) -> None:
        self.pid = int(pid if pid is not None else os.getpid())
        self.process_name = process_name
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._registered_tids: set[int] = set()
        self._pending_cuda_events: list[tuple[str, str, int, Any, Any, dict[str, Any], int]] = []
        self._append_metadata_event("process_name", tid=0, args={"name": process_name})
        self.register_thread("main")
        self.register_thread("gpu_stream_0", tid=self._GPU_TID)

    def _append_metadata_event(self, name: str, *, tid: int, args: dict[str, Any]) -> None:
        self._events.append(
            {
                "name": name,
                "ph": "M",
                "pid": self.pid,
                "tid": tid,
                "args": args,
            }
        )

    def register_thread(self, name: str, *, tid: int | None = None) -> int:
        resolved_tid = int(threading.get_ident() if tid is None else tid)
        with self._lock:
            if resolved_tid not in self._registered_tids:
                self._registered_tids.add(resolved_tid)
                self._append_metadata_event("thread_name", tid=resolved_tid, args={"name": name})
        return resolved_tid

    def add_slice(
        self,
        name: str,
        *,
        category: str,
        start_ns: int,
        end_ns: int,
        tid: int | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        resolved_tid = self.register_thread(category if tid is None else str(tid), tid=tid)
        event = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": _to_us(start_ns),
            "dur": _to_us(max(end_ns - start_ns, 0)),
            "pid": self.pid,
            "tid": resolved_tid,
            "args": args or {},
        }
        with self._lock:
            self._events.append(event)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        category: str,
        tid: int | None = None,
        args: dict[str, Any] | None = None,
    ):
        start_ns = time.perf_counter_ns()
        try:
            yield start_ns
        finally:
            self.add_slice(
                name,
                category=category,
                start_ns=start_ns,
                end_ns=time.perf_counter_ns(),
                tid=tid,
                args=args,
            )

    def add_counter(
        self,
        name: str,
        value: int | float,
        *,
        category: str,
        timestamp_ns: int | None = None,
        tid: int | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        resolved_tid = self.register_thread(category if tid is None else str(tid), tid=tid)
        payload = {"value": value}
        if args:
            payload.update(args)
        event = {
            "name": name,
            "cat": category,
            "ph": "C",
            "ts": _to_us(timestamp_ns if timestamp_ns is not None else time.perf_counter_ns()),
            "pid": self.pid,
            "tid": resolved_tid,
            "args": payload,
        }
        with self._lock:
            self._events.append(event)

    def add_cuda_pending_span(
        self,
        name: str,
        *,
        category: str,
        cpu_begin_ns: int,
        start_event: Any,
        end_event: Any,
        args: dict[str, Any] | None = None,
        tid: int | None = None,
    ) -> None:
        resolved_tid = self._GPU_TID if tid is None else int(tid)
        self.register_thread("gpu_stream_0", tid=resolved_tid)
        with self._lock:
            self._pending_cuda_events.append(
                (name, category, cpu_begin_ns, start_event, end_event, args or {}, resolved_tid)
            )

    def flush_cuda_pending(self) -> None:
        pending: list[tuple[str, str, int, Any, Any, dict[str, Any], int]]
        with self._lock:
            pending = self._pending_cuda_events
            self._pending_cuda_events = []
        for name, category, cpu_begin_ns, start_event, end_event, args, tid in pending:
            end_event.synchronize()
            gpu_dur_us = float(start_event.elapsed_time(end_event) * 1000.0)
            event = {
                "name": name,
                "cat": category,
                "ph": "X",
                "ts": _to_us(cpu_begin_ns),
                "dur": gpu_dur_us,
                "pid": self.pid,
                "tid": tid,
                "args": {**args, "gpu_dur_us": gpu_dur_us},
            }
            with self._lock:
                self._events.append(event)

    def extend(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self._lock:
            self._events.extend(events)

    def drain_events(self) -> list[dict[str, Any]]:
        self.flush_cuda_pending()
        with self._lock:
            metadata = [event for event in self._events if event.get("ph") == "M"]
            others = [event for event in self._events if event.get("ph") != "M"]
            self._events = metadata
        return others

    def write_json(self, output_path: str | Path) -> Path:
        self.flush_cuda_pending()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            events = list(self._events)
        path.write_text(json.dumps({"traceEvents": events}, ensure_ascii=False), encoding="utf-8")
        return path
