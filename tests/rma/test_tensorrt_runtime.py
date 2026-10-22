"""Bridge protocol and cleanup tests using an engine-independent subprocess."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from unilab_rma.tensorrt_runtime import TensorRTWorker

FAKE_WORKER = """
from pathlib import Path
import struct, sys, time
mode = Path(sys.argv[2]).read_text()
if mode == 'startup_failure':
    raise SystemExit(2)
magic = b'BAD!' if mode == 'bad_header' else b'RMA1'
sys.stdout.buffer.write(struct.pack('<4sIII',magic,49,50,12))
sys.stdout.buffer.flush()
if mode == 'bad_header':
    raise SystemExit(0)
while True:
    frame = sys.stdin.buffer.read((49+2450)*4)
    if not frame:
        break
    if len(frame) != (49+2450)*4:
        raise SystemExit(3)
    values = struct.unpack('<2499f',frame)
    action = [values[i]+values[49+2450-49+i] for i in range(12)]
    if mode == 'stall':
        time.sleep(60)
    if mode == 'nan':
        action[0] = float('nan')
    encoded = struct.pack('<12f',*action)
    if mode == 'short_output':
        sys.stdout.buffer.write(encoded[:4])
        sys.stdout.buffer.flush()
        break
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
"""


@unittest.skipUnless(os.name == "posix", "The C++ bridge targets Linux/WSL pipes")
class TensorRTWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.worker = self.folder / "fake_worker"
        self.worker.write_text("#!" + sys.executable + "\n" + FAKE_WORKER)
        self.worker.chmod(0o755)
        self.engine = self.folder / "fake.engine"
        self.cfg = SimpleNamespace(obs_dim=49, history_length=50, action_dim=12)

    def start(self, mode="echo", **kwargs):
        self.engine.write_text(mode)
        runtime = TensorRTWorker(self.worker, self.engine, self.cfg, **kwargs)
        self.addCleanup(runtime.close)
        return runtime

    def test_multiple_environments_preserve_observation_and_history_order(self):
        runtime = self.start()
        process = runtime._process
        obs = torch.arange(98, dtype=torch.float32).reshape(2, 49)
        history = torch.zeros(2, 50, 49)
        history[:, -1, :] = -3 * obs
        actual = runtime.infer(obs, history)
        self.assertTrue(torch.equal(actual, -2 * obs[:, :12]))
        history[:, -1, :] = obs + 1
        self.assertTrue(torch.equal(runtime.infer(obs, history), 2 * obs[:, :12] + 1))
        self.assertIs(runtime._process, process)
        runtime.close()
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdout.closed and process.stdin.closed)
        runtime.close()  # Repeated cleanup is safe.
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.infer(obs, history)

    def test_bad_handshake_and_startup_failure_are_reported(self):
        for mode, message in [("bad_header", "protocol"), ("startup_failure", "closed output")]:
            with self.subTest(mode=mode):
                self.engine.write_text(mode)
                with self.assertRaisesRegex(RuntimeError, message):
                    TensorRTWorker(self.worker, self.engine, self.cfg)

    def test_truncated_action_does_not_hang(self):
        runtime = self.start("short_output")
        with self.assertRaisesRegex(RuntimeError, "closed output"):
            runtime.infer(torch.zeros(1, 49), torch.zeros(1, 50, 49))

    def test_nonfinite_output_is_rejected(self):
        runtime = self.start("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.infer(torch.zeros(1, 49), torch.zeros(1, 50, 49))

    def test_stalled_worker_times_out_and_can_be_stopped(self):
        runtime = self.start("stall", timeout_s=0.5)
        with self.assertRaises(TimeoutError):
            runtime.infer(torch.zeros(1, 49), torch.zeros(1, 50, 49))
        runtime.close()
        self.assertIsNotNone(runtime._process.poll())

    def test_invalid_inputs_are_rejected_before_send(self):
        runtime = self.start()
        with self.assertRaisesRegex(ValueError, "observation shape"):
            runtime.infer(torch.zeros(1, 48), torch.zeros(1, 50, 49))
        with self.assertRaisesRegex(ValueError, "history shape"):
            runtime.infer(torch.zeros(2, 49), torch.zeros(1, 50, 49))
        with self.assertRaisesRegex(ValueError, "finite"):
            runtime.infer(torch.full((1, 49), float("nan")), torch.zeros(1, 50, 49))


if __name__ == "__main__":
    unittest.main()
