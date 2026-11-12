"""CUDA graph capture of the fixed-shape autoreset segment.

``TorchEnv`` with ``masked_autoreset`` and ``persistent_step_buffers`` runs its
autoreset as one fixed-shape body (``TorchEnv._masked_autoreset_body``): every
tensor it touches has a fixed address, nothing in it reads the device from the
host, and host bookkeeping is declared through ``record_host_transition``.
:class:`AutoresetGraph` captures that body once and replays it afterwards.

Capture protocol (all public, no object-graph inspection):

1. ``warmup`` eager executions allocate every lazily created buffer, scratch
   and verdict flag outside the graph.
2. The body runs inside ``torch.cuda.graph`` on a side stream. Within it, the
   backend's ``torch_graph_capture_scope`` routes simulator kernels onto the
   capture stream and returns an ``ExternalGraphCapture`` handle, which the env
   binds so that env/manager host transitions record on the same handle as the
   backend's own.
3. Capture records device work without executing it, so the graph is replayed
   once immediately; the host transitions already ran during capture.
4. Every later call checks the backend's replay guards, replays the graph and
   re-applies the recorded host transitions in recording order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from unisim.backend.graph_capture import ExternalGraphCapture, GraphCaptureError


class AutoresetGraph:
    """Capture-once, replay-always runner for a fixed-shape autoreset body."""

    def __init__(
        self,
        *,
        backend: Any,
        device: torch.device,
        body: Callable[[], None],
        bind_capture: Callable[[ExternalGraphCapture | None], None],
        generators: Sequence[torch.Generator] = (),
        warmup: int = 2,
        capture_error_mode: str = "thread_local",
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"AutoresetGraph requires a CUDA device, got {device}")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError(f"AutoresetGraph warmup must be an integer >= 1, got {warmup!r}")
        scope = getattr(backend, "torch_graph_capture_scope", None)
        if not callable(scope):
            raise NotImplementedError(
                f"{type(backend).__name__} does not implement torch_graph_capture_scope"
            )
        for generator in generators:
            if generator.device.type != "cuda":
                raise ValueError(
                    f"AutoresetGraph generators must be CUDA generators, got {generator.device}"
                )
        self._backend = backend
        self._device = device
        self._body = body
        self._bind_capture = bind_capture
        self._generators = tuple(generators)
        self._warmup = warmup
        self._capture_error_mode = capture_error_mode
        self._eager_calls = 0
        self._graph: torch.cuda.CUDAGraph | None = None
        self._capture: ExternalGraphCapture | None = None
        self.replays = 0

    @property
    def captured(self) -> bool:
        return self._graph is not None

    @property
    def host_transition_names(self) -> tuple[str, ...]:
        """Names of the host transitions replayed after every graph replay."""
        return () if self._capture is None else self._capture.host_transition_names

    def run(self) -> None:
        """Execute the body: eagerly during warmup, then by graph replay."""
        if self._graph is not None:
            capture = self._capture
            assert capture is not None
            capture.check_replay_guards()
            self._graph.replay()
            capture.replay_host_transitions()
            self.replays += 1
            return
        if self._eager_calls < self._warmup:
            self._eager_calls += 1
            self._body()
            return
        self._capture_and_run()

    def _capture_and_run(self) -> None:
        current = torch.cuda.current_stream(self._device)
        stream = torch.cuda.Stream(device=self._device)
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        for generator in self._generators:
            graph.register_generator_state(generator)
        capture: ExternalGraphCapture | None = None
        try:
            with torch.cuda.graph(
                graph, stream=stream, capture_error_mode=self._capture_error_mode
            ):
                with self._backend.torch_graph_capture_scope(
                    stream, capture_error_mode=self._capture_error_mode
                ) as capture:
                    self._bind_capture(capture)
                    try:
                        self._body()
                    finally:
                        self._bind_capture(None)
        except GraphCaptureError:
            raise
        except Exception as exc:
            raise GraphCaptureError(f"autoreset graph capture failed: {exc}") from exc
        current.wait_stream(stream)
        assert capture is not None
        self._graph = graph
        self._capture = capture
        # Capture recorded the device work without executing it; the host
        # transitions already ran once during capture.
        graph.replay()
        self.replays += 1


__all__ = ["AutoresetGraph"]
