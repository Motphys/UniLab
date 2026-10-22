"""RMA reset labels recorded from the SAME samples written to simulation.

Only manager reset-transaction backends are supported; no guessed fallback labels.
Friction is the selected robot geoms' sliding coefficient, not an effective
robot/terrain contact coefficient (contact mixing remains backend-specific).
"""

import numpy as np
import torch

from unilab.envs.mdp.events import GeomFriction, RandomizeRigidBodyMass, resolve_env_ids


def _labels(env):
    if not hasattr(env, "_rma_privileged"):
        # Initialization defaults only, overwritten by the reset events.
        env._rma_privileged = np.tile(np.array([1.0, 0.0]), (env.num_envs, 1))
        env._rma_label_valid = np.zeros((env.num_envs, 2), dtype=bool)
    return env._rma_privileged


def privileged_parameters(env):
    return torch.as_tensor(_labels(env), device=env.device, dtype=torch.float32)


class RMAFriction(GeomFriction):
    def _write(self, values, env_ids):
        coefficients = values[..., 0]
        if not np.allclose(coefficients, coefficients[:, :1]):
            raise ValueError(
                "RMA scalar friction requires shared_random=true across selected geoms"
            )
        super()._write(values, env_ids)
        _labels(self._env)[env_ids, 0] = coefficients[:, 0]
        self._env._rma_label_valid[env_ids, 0] = True


class RMABaseMass(RandomizeRigidBodyMass):
    def __call__(self, env, env_ids, **kwargs):
        if self._operation != "add" or self._distribution != "uniform" or len(self._body_ids) != 1:
            raise ValueError("RMA load requires one base body, uniform additive mass")
        ids = resolve_env_ids(env, env_ids)
        samples = env.rng.uniform(*self._distribution_params, size=(ids.size, 1))
        defaults = self._default_mass
        if defaults.ndim == 2:
            defaults = defaults[ids]
        values = defaults + samples
        if np.any(values < self._min_mass):
            raise ValueError("Invalid randomized base mass")
        self._entity.write_body_mass_to_sim(
            values, body_ids=self._body_ids, env_ids=ids, term_name="randomize_rigid_body_mass"
        )
        _labels(env)[ids, 1] = samples[:, 0]
        env._rma_label_valid[ids, 1] = True
