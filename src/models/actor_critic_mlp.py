import torch
import torch.nn as nn


def _orthogonal_init(m: nn.Module, gain: float = 1.0) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


class Actor(nn.Module):
    """
    Shared-parameter recurrent actor used by all N robots.

    Architecture: MLP feature extractor → GRUCell → linear mean head.
    Actions are sampled from a tanh-squashed Normal distribution:
        z ~ N(mean, exp(log_std)),  a = tanh(z)  ∈ (-1, 1)^action_dim
    """

    def __init__(self, obs_dim: int, action_dim: int = 2,
                 hidden_size: int = 128, gru_hidden: int = 128):
        super().__init__()
        self.gru_hidden = gru_hidden

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.gru       = nn.GRUCell(hidden_size, gru_hidden)
        self.mean_head = nn.Linear(gru_hidden, action_dim)
        self.log_std   = nn.Parameter(torch.zeros(action_dim))

        self.mlp.apply(lambda m: _orthogonal_init(m, gain=1.0))
        _orthogonal_init(self.mean_head, gain=0.01)

    def forward(
        self, obs: torch.Tensor, hx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs : (B, obs_dim)
        hx  : (B, gru_hidden)
        →   mean    (B, action_dim)  — unbounded, used as μ of underlying Gaussian
            log_std (action_dim,)
            hx      (B, gru_hidden)  — updated hidden state
        """
        feat = self.mlp(obs)
        hx   = self.gru(feat, hx)
        mean = self.mean_head(hx)
        return mean, self.log_std, hx

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        return torch.zeros(batch_size, self.gru_hidden, device=device)


class Critic(nn.Module):
    """
    Centralised critic: maps the global state to a scalar value estimate.
    Takes a concatenation of all robot poses, human positions, and coverage grid.
    """

    def __init__(self, state_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.net.apply(lambda m: _orthogonal_init(m, gain=1.0))
        _orthogonal_init(list(self.net.children())[-1], gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state : (B, state_dim) → value : (B, 1)"""
        return self.net(state)
