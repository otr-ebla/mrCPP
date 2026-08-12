"""Flax actor and critic networks.

Orthogonal initialisers require a QR factorisation, which the Metal backend
does not implement. Callers should evaluate `init` on CPU and transfer the
resulting parameters to the accelerator (see `src.utils.jax_device`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


def _dense(features: int, gain: float) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=nn.initializers.orthogonal(gain),
        bias_init=nn.initializers.zeros,
    )


class Actor(nn.Module):
    """
    Shared-parameter recurrent actor used by all N robots.

    Architecture: MLP feature extractor -> GRUCell -> linear mean head.
    Actions are sampled from a tanh-squashed Normal distribution:
        z ~ N(mean, exp(log_std)),  a = tanh(z)  in (-1, 1)^action_dim
    """

    action_dim: int = 2
    hidden_size: int = 128
    gru_hidden: int = 128

    @nn.compact
    def __call__(
        self, obs: jax.Array, hx: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        obs : (B, obs_dim)
        hx  : (B, gru_hidden)
        ->  mean    (B, action_dim)  — unbounded, used as mu of underlying Gaussian
            log_std (action_dim,)
            hx      (B, gru_hidden)  — updated hidden state
        """
        x = nn.tanh(_dense(self.hidden_size, 1.0)(obs))
        x = nn.tanh(_dense(self.hidden_size, 1.0)(x))

        new_hx, y = nn.GRUCell(features=self.gru_hidden)(hx, x)

        mean = _dense(self.action_dim, 0.01)(y)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        return mean, log_std, new_hx

    def init_hidden(self, batch_size: int = 1) -> jax.Array:
        return jnp.zeros((batch_size, self.gru_hidden), jnp.float32)


class Critic(nn.Module):
    """
    Centralised critic: maps the global state to a scalar value estimate.
    Takes a concatenation of all robot poses, human positions, and coverage grid.
    """

    hidden_size: int = 256

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        """state : (B, state_dim) -> value : (B, 1)"""
        x = nn.tanh(_dense(self.hidden_size, 1.0)(state))
        x = nn.tanh(_dense(self.hidden_size, 1.0)(x))
        return _dense(1, 1.0)(x)
