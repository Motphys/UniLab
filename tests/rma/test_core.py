import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from unilab_rma import ActorCriticRMA, AdaptationModule, AdaptationTrainer, HistoryBuffer, RMAConfig
from unilab_rma.deployment import DeploymentPolicy
from unilab_rma.env_adapter import UniLabRMAEnv
from unilab_rma.ppo import gae
from unilab_rma.runner import train_student, train_teacher
from unilab_rma.scene_contract import expanded_xml, prepare_scene

torch.set_num_threads(1)


class SyntheticEnv:
    """Synthetic tensors ONLY, not a Go2 simulation or performance benchmark."""

    num_envs = 2

    def reset(self):
        self.t = 0
        return self.obs()

    def obs(self):
        return (
            torch.randn(2, 49) * 0.1,
            torch.randn(2, 52) * 0.1,
            torch.tensor([[0.5, 1.0], [1.0, 2.0]]),
        )

    def step(self, action):
        self.t += 1
        terminal = self.obs()
        done = torch.tensor([self.t % 3 == 0, self.t % 4 == 0])
        return (
            self.obs(),
            -action.square().mean(-1),
            done,
            torch.zeros(2, dtype=torch.bool),
            terminal,
        )


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.cfg = RMAConfig()

    def test_shapes_and_log_probability(self):
        m = ActorCriticRMA(self.cfg)
        a, raw, logp = m.sample(torch.zeros(3, 49), torch.ones(3, 2))
        self.assertEqual(a.shape, (3, 12))
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue((a.abs() <= 1).all())
        self.assertTrue(
            torch.isfinite(
                m.log_prob(
                    m.distribution(torch.zeros(3, 49), torch.ones(3, 2)),
                    torch.full_like(raw, 100.0),
                )
            ).all()
        )

    def test_history_reset(self):
        h = HistoryBuffer(2, self.cfg)
        h.push(torch.ones(2, 49))
        h.push(torch.full((2, 49), 2.0), torch.tensor([True, False]))
        self.assertEqual(h.data[0, :-1].abs().sum().item(), 0)
        self.assertTrue((h.data[1, -2] == 1).all())
        self.assertTrue((h.data[:, -1] == 2).all())

    def test_timeout_and_termination(self):
        r = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
        v = torch.zeros_like(r)
        nxt = torch.full_like(r, 10.0)
        term = torch.tensor([[True, False], [False, False]])
        trunc = torch.tensor([[False, True], [False, False]])
        a, _ = gae(r, v, nxt, term, trunc)
        self.assertAlmostEqual(a[0, 0].item(), 1.0)
        self.assertAlmostEqual(a[0, 1].item(), 10.9, places=5)

    def test_supervision_freezes_teacher(self):
        teacher, student = ActorCriticRMA(self.cfg), AdaptationModule(self.cfg)
        saved = {k: v.clone() for k, v in teacher.state_dict().items()}
        trainer = AdaptationTrainer(teacher, student, capacity=8)
        history = torch.randn(2, 50, 49)
        trainer.add(history, torch.ones(2, 2))
        with torch.no_grad():
            initial = (student(history) - teacher.encoder(torch.ones(2, 2))).square().mean().item()
        for _ in range(50):
            trainer.update(batch_size=2, updates=2)
        with torch.no_grad():
            final = (student(history) - teacher.encoder(torch.ones(2, 2))).square().mean().item()
        self.assertLess(final, initial)
        for key, value in teacher.state_dict().items():
            self.assertTrue(torch.equal(value, saved[key]))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_rollout_update_and_checkpoint(self):
        teacher, student = ActorCriticRMA(self.cfg), AdaptationModule(self.cfg)
        before = teacher.actor.net[0].weight.detach().clone()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "teacher.pt"
            train_teacher(SyntheticEnv(), teacher, student, 1, p, horizon=4)
            self.assertFalse(torch.equal(before, teacher.actor.net[0].weight))
            train_student(SyntheticEnv(), teacher, student, 1, Path(d) / "student.pt", horizon=4)
            saved = torch.load(Path(d) / "student.pt", weights_only=True)
            self.assertEqual(saved["phase"], "student")
            deploy = DeploymentPolicy(self.cfg)
            deploy.actor.load_state_dict(teacher.actor.state_dict())
            deploy.adaptation.load_state_dict(student.state_dict())
            obs, h = torch.randn(2, 49), torch.randn(2, 50, 49)
            torch.testing.assert_close(deploy(obs, h), teacher.actor(obs, student(h)).tanh())
            self.assertFalse(any("encoder" in k or "critic" in k for k in deploy.state_dict()))

    def test_scene_names_from_xml_includes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "robot.xml").write_text(
                '<mujoco><body name="base"><geom name="foot"/></body></mujoco>'
            )
            (p / "scene.xml").write_text(
                '<mujoco><worldbody><include file="robot.xml"/></worldbody></mujoco>'
            )
            override = {"scene": {"model_file": "scene.xml", "entities": {"robot": {}}}}
            prepare_scene(override, p)
            self.assertEqual(override["scene"]["entities"]["robot"]["geom_names"], ["foot"])

    def test_adapter_preserves_pre_reset_state(self):
        class Raw:
            _autoreset = False
            num_envs = 2
            device = "cpu"
            _rma_label_valid = np.ones((2, 2), dtype=bool)

            def __init__(self):
                self.data = {
                    k: torch.zeros(2, n)
                    for k, n in [("policy", 49), ("critic", 52), ("privileged", 2)]
                }

            def step(self, action):
                for x in self.data.values():
                    x.fill_(7.0)
                return SimpleNamespace(
                    obs=self.data,
                    reward=torch.ones(2),
                    terminated=torch.tensor([False, False]),
                    truncated=torch.tensor([True, False]),
                )

            def reset(self, ids):
                for x in self.data.values():
                    x[ids] = 0
                return self.data, {}

            def get_observations(self):
                return self.data

        env = UniLabRMAEnv(Raw(), self.cfg)
        current, _, term, trunc, terminal = env.step(torch.zeros(2, 12))
        self.assertEqual(terminal[0][0, 0].item(), 7.0)
        self.assertEqual(current[0][0, 0].item(), 0.0)
        self.assertEqual(current[0][1, 0].item(), 7.0)

    def test_adapter_public_mapping_and_label_snapshot(self):
        raw = SimpleNamespace(
            _autoreset=False,
            num_envs=2,
            _rma_label_valid=np.ones((2, 2), dtype=bool),
            _rma_privileged=np.array([[0.5, 1.0], [1.0, 2.0]]),
        )
        env = UniLabRMAEnv(raw, self.cfg)
        obs, critic, priv = env.read({"obs": torch.zeros(2, 49), "critic": torch.ones(2, 52)})
        self.assertEqual(obs.shape, (2, 49))
        self.assertEqual(critic.shape, (2, 52))
        raw._rma_privileged[0] = 0
        self.assertEqual(priv[0, 1].item(), 1.0)
        with self.assertRaisesRegex(KeyError, "critic_observation_group"):
            env.read({"obs": torch.zeros(2, 49)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
