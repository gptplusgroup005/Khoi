import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        obs = obs.float()
        feat = self.feature(obs.float())
        logits = self.actor(feat)
        if obs.shape[-1] >= 15 and logits.shape[-1] == 4:
            logits = logits - 40.0 * obs[:, 11:15]
        value = self.critic(feat).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(action)

        return action, logp, None, value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        actions = actions.long().view(-1)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()

        return logp, value, entropy