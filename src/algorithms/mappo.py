from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RolloutBuffer:
    """
    Fixed-length (T steps, E parallel envs, N agents) experience buffer.

    All tensors live on the target device.  The buffer is filled one step at a
    time via insert(), then compute_returns() closes the GAE computation and
    resets the write pointer so the buffer can be reused.

    Shapes
    ------
    obs       : (T, E, N, obs_dim)
    states    : (T, E, state_dim)
    actions   : (T, E, N, action_dim)
    log_probs : (T, E, N)
    rewards   : (T, E, N)
    values    : (T+1, E)     — +1 row for the bootstrap value
    dones     : (T, E)       — 1.0 when the episode terminates (hard collision)
    hx        : (T, E, N, gru_hidden)
    """

    def __init__(
        self,
        T: int,
        E: int,
        N: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        gru_hidden: int,
        device: torch.device,
    ):
        self.T, self.E, self.N = T, E, N
        self.device = device

        self.obs       = torch.zeros(T, E, N, obs_dim,    device=device)
        self.states    = torch.zeros(T, E, state_dim,     device=device)
        self.actions   = torch.zeros(T, E, N, action_dim, device=device)
        self.log_probs = torch.zeros(T, E, N,             device=device)
        self.rewards   = torch.zeros(T, E, N,             device=device)
        self.values    = torch.zeros(T + 1, E,            device=device)   # +1 for bootstrap
        self.dones     = torch.zeros(T, E,                device=device)
        self.hx        = torch.zeros(T, E, N, gru_hidden, device=device)
        self.ptr       = 0

        self.advantages: torch.Tensor | None = None
        self.returns:    torch.Tensor | None = None

    def insert(
        self,
        obs:       torch.Tensor,   # (E, N, obs_dim)
        states:    torch.Tensor,   # (E, state_dim)
        actions:   torch.Tensor,   # (E, N, action_dim)
        log_probs: torch.Tensor,   # (E, N)
        rewards:   torch.Tensor,   # (E, N)
        values:    torch.Tensor,   # (E,)
        dones:     torch.Tensor,   # (E,)
        hx:        torch.Tensor,   # (E, N, gru_hidden)
    ) -> None:
        t = self.ptr
        self.obs[t]       = obs
        self.states[t]    = states
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = rewards
        self.values[t]    = values
        self.dones[t]     = dones
        self.hx[t]        = hx
        self.ptr += 1

    def compute_returns(
        self,
        last_values: torch.Tensor,   # (E,)
        gamma:       float,
        gae_lambda:  float,
    ) -> None:
        """
        GAE-λ advantage and discounted return computation, independent per env.

        last_values — V(s_T) bootstrap value for each env after the last stored step.
        dones       — 1 only on hard termination (collision); truncation bootstraps normally.
        """
        self.values[self.ptr] = last_values   # (E,)

        advantages = torch.zeros(self.T, self.E, device=self.device)
        gae        = torch.zeros(self.E,         device=self.device)

        for t in reversed(range(self.T)):
            mask   = 1.0 - self.dones[t]                            # (E,)
            team_r = self.rewards[t].mean(dim=-1)                    # (E,) — mean over N agents
            delta  = team_r + gamma * self.values[t + 1] * mask - self.values[t]
            gae    = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

        self.returns    = advantages + self.values[: self.T]   # (T, E)
        self.advantages = advantages                            # (T, E)
        self.ptr        = 0


class MAPPO:
    """
    Multi-Agent PPO with a centralised critic and parameter-shared actor.

    Supports E parallel environments: all actor and critic forward passes are
    batched across (E × N) agent-slots in a single call, so data collection is
    as fast as a single vectorised network pass.
    """

    def __init__(
        self,
        actor,
        critic,
        config: dict,
        device: str = 'cpu',
    ):
        self.actor  = actor.to(device)
        self.critic = critic.to(device)
        self.device = torch.device(device)

        self.actor_opt = torch.optim.Adam(
            actor.parameters(),  lr=config.get('lr_actor',  3e-4), eps=1e-5
        )
        self.critic_opt = torch.optim.Adam(
            critic.parameters(), lr=config.get('lr_critic', 1e-3), eps=1e-5
        )

        self.clip_eps      = config.get('clip_eps',      0.2)
        self.entropy_coef  = config.get('entropy_coef',  0.01)
        self.max_grad_norm = config.get('max_grad_norm', 10.0)
        self.n_epochs      = config.get('n_epochs',      10)
        self.gamma         = config.get('gamma',         0.99)
        self.gae_lambda    = config.get('gae_lambda',    0.95)

    # ------------------------------------------------------------------
    # Action sampling (no gradients, used during rollout collection)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_actions(
        self,
        obs_np:   np.ndarray,   # (E, N, obs_dim)
        state_np: np.ndarray,   # (E, state_dim)
        hx:       torch.Tensor, # (E, N, gru_hidden)  on device
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        actions   : (E, N, action_dim)  numpy, squashed to (-1, 1)
        log_probs : (E, N)              tensor (CPU)
        values    : (E,)                tensor (CPU)
        new_hx    : (E, N, gru_hidden)  tensor (device)
        """
        E, N = obs_np.shape[:2]

        obs   = torch.from_numpy(obs_np).float().to(self.device)    # (E, N, obs_dim)
        state = torch.from_numpy(state_np).float().to(self.device)  # (E, state_dim)

        # Flatten E×N into one batch for the shared actor
        obs_f = obs.reshape(E * N, -1)
        hx_f  = hx.reshape(E * N, -1)

        mean, log_std, new_hx_f = self.actor(obs_f, hx_f)
        std  = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        z    = dist.sample()
        a    = torch.tanh(z)
        lp   = (dist.log_prob(z) - torch.log(1.0 - a.pow(2) + 1e-6)).sum(-1)  # (E*N,)

        a      = a.reshape(E, N, -1)       # (E, N, action_dim)
        lp     = lp.reshape(E, N)          # (E, N)
        new_hx = new_hx_f.reshape(E, N, -1)

        # Centralised critic: one value per env
        values = self.critic(state).squeeze(-1)   # (E,)

        return a.cpu().numpy(), lp.cpu(), values.cpu(), new_hx

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> dict:
        T, E, N = buffer.T, buffer.E, buffer.N

        # Flatten (T, E, N, ...) → (T*E*N, ...) — each (t, e, i) triple is
        # processed independently (truncated-BPTT-length-1 approximation).
        obs_f  = buffer.obs.reshape(T * E * N, -1)
        act_f  = buffer.actions.reshape(T * E * N, -1)
        olp_f  = buffer.log_probs.reshape(T * E * N)
        hx_f   = buffer.hx.reshape(T * E * N, -1)

        # Shared advantage: one value per (t, e), broadcast over N agents
        adv = buffer.advantages                             # (T, E)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_f = adv.unsqueeze(-1).expand(T, E, N).reshape(T * E * N)

        ret = buffer.returns                                # (T, E)

        a_losses, c_losses, ents = [], [], []

        for _ in range(self.n_epochs):
            # ---- Actor ----
            mean, log_std, _ = self.actor(obs_f, hx_f)
            std  = log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            z    = torch.atanh(act_f.clamp(-1 + 1e-6, 1 - 1e-6))
            nlp  = (dist.log_prob(z) - torch.log(1.0 - act_f.pow(2) + 1e-6)).sum(-1)

            ratio = (nlp - olp_f).exp()
            surr1 = ratio * adv_f
            surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_f
            a_loss = -torch.min(surr1, surr2).mean()
            ent    = dist.entropy().sum(-1).mean()

            self.actor_opt.zero_grad()
            (a_loss - self.entropy_coef * ent).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

            # ---- Critic ----
            states_f = buffer.states.reshape(T * E, -1)              # (T*E, state_dim)
            vals     = self.critic(states_f).squeeze(-1).reshape(T, E)  # (T, E)
            c_loss   = 0.5 * ((vals - ret) ** 2).mean()

            self.critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.step()

            a_losses.append(a_loss.item())
            c_losses.append(c_loss.item())
            ents.append(ent.item())

        return {
            'actor_loss':  float(np.mean(a_losses)),
            'critic_loss': float(np.mean(c_losses)),
            'entropy':     float(np.mean(ents)),
        }
