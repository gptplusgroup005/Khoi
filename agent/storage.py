import torch

class RolloutStorage:
    def __init__(self, num_steps: int, obs_dim: int, num_envs: int, device):
        self.device = device
        self.num_steps = num_steps
        self.step = 0
        self.num_envs = num_envs
        self.obs_dim = obs_dim

        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.logp = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)

        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

    def clear(self):
        self.step = 0

    def add(self, obs, action, logp, reward, done, value):
        if self.step >= self.num_steps:
            raise RuntimeError("RolloutStorage overflow")

        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(action.view(-1).long())
        self.logp[self.step].copy_(logp)
        self.rewards[self.step].copy_(reward)
        self.dones[self.step].copy_(done)
        self.values[self.step].copy_(value)

        self.step += 1

    def compute_returns(self, last_value, gamma=0.99, gae_lambda=0.95):

        gae = torch.zeros(self.num_envs, device=self.device)
        temp_values = torch.cat([self.values[:self.step], last_value.unsqueeze(0)], dim=0)
        
        for t in reversed(range(self.step)):
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * temp_values[t + 1] * next_non_terminal - temp_values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = self.advantages[t] + self.values[t]

        adv = self.advantages[:self.step]
        self.advantages[:self.step] = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.advantages[:self.step] = torch.clamp(self.advantages[:self.step], -5.0, 5.0)

    def mini_batches(self, num_mini_batches: int):
        batch_size = self.step * self.num_envs
        mini_batch_size = max(1, batch_size // num_mini_batches)

        obs = self.obs[:self.step].reshape(batch_size, self.obs_dim)
        actions = self.actions[:self.step].reshape(batch_size)
        logp = self.logp[:self.step].reshape(batch_size)
        returns = self.returns[:self.step].reshape(batch_size)
        adv = self.advantages[:self.step].reshape(batch_size)
        values = self.values[:self.step].reshape(batch_size)

        indices = torch.randperm(batch_size, device=self.device)

        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            mb = indices[start:end]

            yield (
                obs[mb],
                actions[mb],
                logp[mb],
                returns[mb],
                adv[mb],
                values[mb],
                
            )