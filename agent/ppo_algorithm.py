import torch
import torch.nn as nn

class PpoAlgorithm:
    def __init__(self, policy, *,
                 learning_rate=1e-4,
                 clip_param=0.2,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 num_learning_epochs=4,
                 num_mini_batches=8,
                 use_clipped_value_loss=True,
                 max_grad_norm=0.5):
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_grad_norm = max_grad_norm

    def update(self, storage):
        total_pi, total_v, total_ent, total_kl, updates = 0.0, 0.0, 0.0, 0.0, 0

        device = next(self.policy.parameters()).device
        for _ in range(self.num_learning_epochs):
            for obs, actions, old_logp, returns, advantages, old_values in storage.mini_batches(self.num_mini_batches):
                
                obs = obs.to(device)
                actions = actions.to(device)
                old_logp = old_logp.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                old_values = old_values.to(device)
                logp, values, entropy = self.policy.evaluate(obs, actions)

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    v_clipped = old_values + torch.clamp(values - old_values, -self.clip_param, self.clip_param)
                    value_loss = torch.max((returns - values).pow(2), (returns - v_clipped).pow(2)).mean()
                else:
                    value_loss = (returns - values).pow(2).mean()

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logp - logp).mean().abs()

                total_pi += float(policy_loss.item())
                total_v += float(value_loss.item())
                total_ent += float(entropy.mean().item())
                total_kl += float(approx_kl.item())
                updates += 1

        return {
            "policy_loss": total_pi / max(1, updates),
            "value_loss": total_v / max(1, updates),
            "entropy": total_ent / max(1, updates),
            "approx_kl": total_kl / max(1, updates),
        }
