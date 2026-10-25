from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scripts.benchmark.physics import benchmark_g1_physics_throughput as benchmark


def test_default_matrix_covers_requested_backends_and_env_counts() -> None:
    args = benchmark.build_parser().parse_args([])

    assert benchmark._parse_backends(args.backends) == ["genesis", "mjbatch", "motrixsim"]
    assert benchmark._parse_backends("mjwarp") == ["mjwarp"]
    assert benchmark._parse_backends("newton") == ["newton"]
    assert benchmark.parse_sizes(args.batch_sizes) == [1024, 2048, 4096, 8192]
    assert benchmark.TASK == "g1_walk_flat"


def test_genesis_selects_native_gpu_and_rejects_unsupported_platforms() -> None:
    assert benchmark._genesis_device("darwin", "arm64") == "metal"
    assert benchmark._genesis_device("linux", "x86_64", hip=None) == "cuda"
    assert benchmark._genesis_device("linux", "x86_64", hip="7.2") == "amdgpu"
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        benchmark._genesis_device("darwin", "x86_64")
    with pytest.raises(RuntimeError, match="Metal or Linux CUDA/ROCm"):
        benchmark._genesis_device("win32", "AMD64")


def test_genesis_synchronization_uses_selected_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(benchmark.torch.mps, "synchronize", lambda: calls.append("mps"))
    monkeypatch.setattr(benchmark.torch.cuda, "synchronize", lambda: calls.append("cuda"))

    benchmark._genesis_synchronize("metal")()
    benchmark._genesis_synchronize("cuda")()
    benchmark._genesis_synchronize("amdgpu")()

    assert calls == ["mps", "cuda", "cuda"]


def test_mjbatch_initializes_stand_state_and_times_only_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = {}
    steps = []

    class Batch:
        def __init__(self, model, batch_size, num_threads):
            assert batch_size == 2
            assert num_threads == 3

        def bind(self, name, dtype):
            bindings[name] = np.empty((2, 3), dtype=dtype)
            return bindings[name]

        def step(self, nstep):
            steps.append(nstep)

    monkeypatch.setattr(benchmark, "mjbatch", SimpleNamespace(Batch=Batch))
    run = benchmark._build_mjbatch(
        object(),
        np.arange(3, dtype=np.float64),
        np.arange(3, dtype=np.float64) + 10,
        np.arange(3, dtype=np.float64) + 20,
        batch_size=2,
        nstep=20,
        nthread=3,
    )

    run.step()
    run.synchronize()

    assert steps == [20]
    assert set(bindings) == {"qpos", "qvel", "ctrl"}
    np.testing.assert_array_equal(bindings["ctrl"][0], [20, 21, 22])


def test_genesis_initializes_metal_scene_and_stand_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    states = {}

    class Scene:
        def __init__(self, **kwargs):
            calls.append(("scene", kwargs))

        def add_entity(self, morph):
            calls.append(("entity", morph))
            return self

        def build(self, n_envs):
            calls.append(("build", n_envs))

        def set_qpos(self, qpos):
            states["qpos"] = qpos

        def control_dofs_position(self, ctrl, dofs_idx_local):
            states["ctrl"] = ctrl
            states["dofs_idx"] = dofs_idx_local

        def step(self):
            calls.append(("step",))

        def destroy(self):
            calls.append(("scene_destroy",))

    fake_gs = SimpleNamespace(
        metal="metal-token",
        backend="metal-token",
        destroy=lambda: calls.append(("destroy",)),
        init=lambda **kwargs: calls.append(("init", kwargs)),
        Scene=Scene,
        options=SimpleNamespace(
            RigidOptions=lambda **kwargs: kwargs,
            ProfilingOptions=lambda **kwargs: kwargs,
        ),
        integrator=SimpleNamespace(implicitfast="implicitfast-token"),
        morphs=SimpleNamespace(MJCF=lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(benchmark, "gs", fake_gs)
    monkeypatch.setattr(benchmark, "_genesis_device", lambda: "metal")
    monkeypatch.setattr(benchmark, "_genesis_synchronize", lambda device: lambda: None)
    model = SimpleNamespace(jnt_dofadr=np.array([6, 7, 8]), actuator_trnid=np.array([[1], [2]]))
    monkeypatch.setattr(
        benchmark,
        "_load_keyframe",
        lambda path: (model, np.array([1, 2]), None, np.array([3, 4])),
    )

    run = benchmark._build_genesis("g1.xml", batch_size=2, nstep=3)
    run.step()
    run.cleanup()

    assert ("init", {"backend": "metal-token", "precision": "32"}) in calls
    scene_options = next(call[1] for call in calls if call[0] == "scene")
    assert scene_options["profiling_options"] == {"show_FPS": False}
    assert calls.count(("step",)) == 3
    np.testing.assert_array_equal(states["qpos"], [[1, 2], [1, 2]])
    np.testing.assert_array_equal(states["ctrl"], [[3, 4], [3, 4]])


def test_genesis_build_failure_destroys_global_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Scene:
        def __init__(self, **kwargs):
            pass

        def add_entity(self, morph):
            return self

        def build(self, n_envs):
            raise RuntimeError("out of memory")

    fake_gs = SimpleNamespace(
        metal="metal-token",
        backend="metal-token",
        destroy=lambda: calls.append("destroy"),
        init=lambda **kwargs: None,
        Scene=Scene,
        options=SimpleNamespace(
            RigidOptions=lambda **kwargs: kwargs,
            ProfilingOptions=lambda **kwargs: kwargs,
        ),
        integrator=SimpleNamespace(implicitfast="implicitfast-token"),
        morphs=SimpleNamespace(MJCF=lambda **kwargs: kwargs),
    )
    monkeypatch.setattr(benchmark, "gs", fake_gs)
    monkeypatch.setattr(benchmark, "_genesis_device", lambda: "metal")

    with pytest.raises(RuntimeError, match="out of memory"):
        benchmark._build_genesis("g1.xml", batch_size=8192, nstep=20)

    assert calls == ["destroy", "destroy"]


def test_motrixsim_initializes_keyframe_controls_and_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    data = SimpleNamespace(actuator_ctrls=None)
    keyframe = SimpleNamespace(
        ctrl=np.array([3, 4], dtype=np.float32),
        apply=lambda value: calls.append(("apply", value)),
    )
    model = SimpleNamespace(
        options=SimpleNamespace(max_iterations=None),
        num_keyframes=1,
        num_actuators=2,
        keyframes=[keyframe],
        forward_kinematic=lambda value: calls.append(("forward", value)),
        step_n=lambda value, nstep: calls.append(("step", value, nstep)),
    )
    fake_mtx = SimpleNamespace(
        load_model=lambda path: model,
        SceneData=lambda model, batch: data,
    )
    monkeypatch.setattr(benchmark, "mtx", fake_mtx)

    run = benchmark._build_motrixsim("g1.xml", batch_size=2, nstep=20, nthread=4)
    run.step()

    assert model.options.max_iterations == 3
    assert calls[:2] == [("apply", data), ("forward", data)]
    assert calls[2] == ("step", data, 20)
    assert run.nthread == 4
    np.testing.assert_array_equal(data.actuator_ctrls, [[3, 4], [3, 4]])


def test_mjwarp_initializes_keyframe_and_synchronizes_step_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Array:
        def __init__(self):
            self.value = None

        def assign(self, value):
            self.value = value

    data = SimpleNamespace(qpos=Array(), qvel=Array(), ctrl=Array())
    fake_mj_warp = SimpleNamespace(
        put_model=lambda model: "warp-model",
        make_data=lambda model, **kwargs: data,
        step=lambda model, value: calls.append(("step", model, value)),
    )
    fake_warp = SimpleNamespace(
        get_device=lambda: SimpleNamespace(is_cuda=True),
        synchronize=lambda: calls.append(("sync",)),
    )
    monkeypatch.setattr(benchmark, "mj_warp", fake_mj_warp)
    monkeypatch.setattr(benchmark, "warp", fake_warp)

    run = benchmark._build_mjwarp(
        object(),
        np.array([1, 2]),
        np.array([3, 4]),
        np.array([5, 6]),
        batch_size=2,
        nstep=2,
    )
    run.step()
    run.synchronize()

    assert calls == [
        ("step", "warp-model", data),
        ("step", "warp-model", data),
        ("sync",),
    ]
    np.testing.assert_array_equal(data.qpos.value, [[1, 2], [1, 2]])
    np.testing.assert_array_equal(data.qvel.value, [[3, 4], [3, 4]])
    np.testing.assert_array_equal(data.ctrl.value, [[5, 6], [5, 6]])


def test_newton_converts_quaternion_and_actuator_targets() -> None:
    model = SimpleNamespace(
        nq=7,
        nv=6,
        nu=2,
        actuator_trnid=np.array([[0], [1]]),
        actuator_biasprm=np.array([[0, 1, 0], [0, 0, 1]]),
        jnt_qposadr=np.array([3, 4]),
        jnt_dofadr=np.array([2, 3]),
    )

    qpos, qvel, target_q, target_qd = benchmark._newton_state_and_targets(
        model,
        np.array([0, 0, 0, 1, 2, 3, 4]),
        np.zeros(6),
        np.array([0.5, -0.5]),
    )

    np.testing.assert_array_equal(qpos, [0, 0, 0, 2, 3, 4, 1])
    np.testing.assert_array_equal(qvel, np.zeros(6))
    np.testing.assert_array_equal(target_q, [0, 0, 0, 0.5, 0, 0, 0])
    np.testing.assert_array_equal(target_qd, [0, 0, 0, -0.5, 0, 0])


def test_benchmark_reports_physics_throughput_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned = []
    run = benchmark.PhysicsRun(
        step=lambda: None,
        synchronize=lambda: None,
        cleanup=lambda: cleaned.append(True),
        device="genesis_metal",
        nthread=0,
    )
    monkeypatch.setattr(benchmark, "_load_keyframe", lambda path: (None, None, None, None))
    monkeypatch.setattr(benchmark, "_build_genesis", lambda *args: run)
    monkeypatch.setattr(benchmark, "bench_callable", lambda *args, **kwargs: [0.5, 1.0])

    record = benchmark._benchmark("genesis", "g1.xml", 1024, 20, 2, 2, 0)

    assert record["avg_time_sec"] == 0.75
    assert record["sps"] == 1024 * 20 / 0.75
    assert record["per_env_sps"] == record["sps"] / 1024
    assert record["device"] == "genesis_metal"
    assert cleaned == [True]


def test_missing_dependency_is_an_explicit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "gs", None)
    monkeypatch.setattr(benchmark, "_GENESIS_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="genesis is unavailable.*genesis extra"):
        benchmark._require_backend("genesis")


@pytest.mark.parametrize("value", ["", "genesis,unknown", "genesis,genesis"])
def test_backend_validation_rejects_invalid_matrices(value: str) -> None:
    with pytest.raises(ValueError):
        benchmark._parse_backends(value)
