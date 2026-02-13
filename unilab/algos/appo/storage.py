import torch

class RolloutStorage:
    """Minimal rollout storage for BOPRL."""
    def __init__(self, num_envs, num_steps_per_env, obs_shape, action_shape, device='cpu'):
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device

        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.actions_log_prob = []
        self.step = 0

    def add_transition(self, transition):
        # Transition is expected to be a dict or object with:
        # observations, actions, rewards, dones, values, actions_log_prob
        # For simplicity in this baseline, we append to lists and stack later
        # Optimization: Pre-allocate tensors if performance is critical
        self.observations.append(transition['observations'].to(self.device))
        self.actions.append(transition['actions'].to(self.device))
        self.rewards.append(transition['rewards'].to(self.device))
        self.dones.append(transition['dones'].to(self.device))
        self.values.append(transition['values'].to(self.device))
        self.actions_log_prob.append(transition['actions_log_prob'].to(self.device))
        self.step += 1

    def get_batch(self):
        # Stack lists into tensors
        return {
            'observations': torch.stack(self.observations),
            'actions': torch.stack(self.actions),
            'rewards': torch.stack(self.rewards),
            'dones': torch.stack(self.dones),
            'values': torch.stack(self.values),
            'actions_log_prob': torch.stack(self.actions_log_prob),
        }
