"""
Multi-Agent PPO in JAX: centralised critic, parameter-shared recurrent actor.

The whole rollout is a `lax.scan` over the vectorised environment, so a full
(T steps x E envs) batch is collected in one device call with no Python in the
inner loop. The PPO update is likewise a `lax.scan` over epochs.

Shapes
------
obs       : (T, E, N, obs_dim)   — already observation-normalised
gstate    : (T, E, state_dim)
action    : (T, E, N, action_dim)
log_prob  : (T, E, N)
reward    : (T, E, N)
value     : (T, E)
term      : (T, E)               — hard termination (collision) only
done      : (T, E)               — terminated OR truncated
hx        : (T, E, N, gru_hidden)

`term` and `done` are stored as float32 0/1 masks rather than bools: jax-metal
0.1.1 returns all-False for any bool-dtype `lax.scan` output, which would
silently erase every episode boundary from the trajectory (see `rollout`).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from src.utils.jax_device import init_device as _pick_init_device

_LOG_2PI = float(np.log(2.0 * np.pi))
_ENTROPY_CONST = 0.5 * float(np.log(2.0 * np.pi * np.e))


# ---------------------------------------------------------------------------
# Observation normalisation (Welford, as a JAX pytree)
# ---------------------------------------------------------------------------

class RunningMeanStd(NamedTuple):
    mean:  jax.Array
    var:   jax.Array
    count: jax.Array


def rms_init(dim: int, eps: float = 1e-4) -> RunningMeanStd:
    return RunningMeanStd(
        jnp.zeros((dim,), jnp.float32),
        jnp.ones((dim,), jnp.float32),
        jnp.float32(eps),
    )


def rms_update(rms: RunningMeanStd, x: jax.Array) -> RunningMeanStd:
    """x : (B, dim) — a flat batch of observations."""
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = x.shape[0]

    total = rms.count + batch_count
    delta = batch_mean - rms.mean
    mean = rms.mean + delta * batch_count / total
    m2 = rms.var * rms.count + batch_var * batch_count \
        + delta ** 2 * rms.count * batch_count / total
    return RunningMeanStd(mean, m2 / total, total)


def rms_normalize(rms: RunningMeanStd, x: jax.Array) -> jax.Array:
    """Normalise x : (..., dim), preserving leading dimensions."""
    return (x - rms.mean) / jnp.sqrt(rms.var + 1e-8)


# ---------------------------------------------------------------------------
# Rollout data
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs:      jax.Array
    gstate:   jax.Array
    action:   jax.Array
    log_prob: jax.Array
    reward:   jax.Array
    value:    jax.Array
    term:     jax.Array   # float32 0/1, not bool — see module docstring
    done:     jax.Array   # float32 0/1, not bool — see module docstring
    hx:       jax.Array
    coverage: jax.Array


class RolloutCarry(NamedTuple):
    """State threaded across rollout steps and between successive updates."""

    env_state: object
    obs:       jax.Array
    gstate:    jax.Array
    hx:        jax.Array
    rms:       RunningMeanStd


def _tanh_normal_log_prob(
    z: jax.Array, mean: jax.Array, std: jax.Array, action: jax.Array
) -> jax.Array:
    """Log density of a tanh-squashed Normal, summed over the action dimension."""
    log_prob = -0.5 * ((z - mean) / std) ** 2 - jnp.log(std) - 0.5 * _LOG_2PI
    log_prob = log_prob - jnp.log(1.0 - action ** 2 + 1e-6)
    return jnp.sum(log_prob, axis=-1)


def compute_gae(
    traj: Transition, last_value: jax.Array, gamma: float, gae_lambda: float
) -> tuple[jax.Array, jax.Array]:
    """
    GAE-lambda advantages and discounted returns, independent per env.

    Masking uses `term` only: a hard termination (collision) stops the
    bootstrap, whereas truncation keeps it because V(s_timeout) is meaningful.
    """
    values = jnp.concatenate([traj.value, last_value[None, :]], axis=0)  # (T+1, E)
    team_reward = jnp.mean(traj.reward, axis=-1)                        # (T, E)
    mask = 1.0 - traj.term.astype(jnp.float32)

    def body(gae, xs):
        reward, value, next_value, m = xs
        delta = reward + gamma * next_value * m - value
        gae = delta + gamma * gae_lambda * m * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        body,
        jnp.zeros_like(last_value),
        (team_reward, values[:-1], values[1:], mask),
        reverse=True,
    )
    return advantages, advantages + values[:-1]


def _apply_gradients(state: TrainState, grads, lr: jax.Array) -> TrainState:
    """Adam step with an explicitly supplied (traceable) learning rate.

    `tx` holds gradient clipping and Adam's moment rescaling; multiplying by
    -lr afterwards is exactly what optax.adam's final scaling does, but keeps
    the rate a traced value so it can be decayed without rebuilding the state.
    """
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    updates = jax.tree_util.tree_map(lambda u: -lr * u, updates)
    return state.replace(
        step=state.step + 1,
        params=optax.apply_updates(state.params, updates),
        opt_state=opt_state,
    )


class MAPPO:
    """
    Multi-Agent PPO with a centralised critic and parameter-shared actor.

    Supports E parallel environments: all actor and critic forward passes are
    batched across (E x N) agent-slots in a single call, so data collection is
    as fast as a single vectorised network pass.
    """

    def __init__(
        self,
        actor,
        critic,
        vec_env,
        config: dict,
        device: jax.Device | None = None,
    ):
        self.actor = actor
        self.critic = critic
        self.env = vec_env

        self.device = device or jax.devices()[0]
        self.init_device = _pick_init_device(self.device)

        self.clip_eps      = float(config.get('clip_eps',      0.2))
        self.entropy_coef  = float(config.get('entropy_coef',  0.01))
        self.max_grad_norm = float(config.get('max_grad_norm', 10.0))
        self.n_epochs      = int(config.get('n_epochs',      10))
        self.gamma         = float(config.get('gamma',         0.99))
        self.gae_lambda    = float(config.get('gae_lambda',    0.95))

        self._update_fn = jax.jit(self._update)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def create_train_states(self, key: jax.Array) -> tuple[TrainState, TrainState]:
        """Initialise parameters (on CPU when the backend lacks QR) and optimisers."""
        actor_key, critic_key = jax.random.split(key)

        dummy_obs = jnp.zeros((1, self.env.obs_dim), jnp.float32)
        dummy_hx = jnp.zeros((1, self.actor.gru_hidden), jnp.float32)
        dummy_state = jnp.zeros((1, self.env.state_dim), jnp.float32)

        with jax.default_device(self.init_device):
            args = jax.device_put(
                (actor_key, critic_key, dummy_obs, dummy_hx, dummy_state),
                self.init_device,
            )
            a_key, c_key, o, h, s = args
            actor_params = self.actor.init(a_key, o, h)
            critic_params = self.critic.init(c_key, s)

        actor_params = jax.device_put(actor_params, self.device)
        critic_params = jax.device_put(critic_params, self.device)

        # Adam's learning rate is applied in _apply_gradients so it stays traceable.
        def make_tx():
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.scale_by_adam(eps=1e-5),
            )

        actor_state = TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=make_tx()
        )
        critic_state = TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=make_tx()
        )
        return actor_state, critic_state

    def init_carry(self, key: jax.Array) -> RolloutCarry:
        env_state, obs, gstate, _ = self.env.reset(key)
        hx = jnp.zeros(
            (self.env.E, self.env.num_robots, self.actor.gru_hidden), jnp.float32
        )
        return RolloutCarry(env_state, obs, gstate, hx, rms_init(self.env.obs_dim))

    # ------------------------------------------------------------------
    # Rollout collection (single device call for T steps)
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0, 4))
    def rollout(
        self,
        actor_params,
        critic_params,
        carry: RolloutCarry,
        num_steps: int,
        key: jax.Array,
    ) -> tuple[RolloutCarry, Transition, jax.Array]:
        """Collect `num_steps` transitions across all E envs.

        Takes raw parameters rather than TrainStates so the optimiser state is
        not passed through the jit boundary (jax-metal rejects the resulting
        signature).

        Returns the updated carry, the stacked trajectory, and the bootstrap
        value V(s_T) for each env.
        """
        e, n = self.env.E, self.env.num_robots

        def step(carry: RolloutCarry, step_key: jax.Array):
            # Statistics are refreshed on the raw observation before it is
            # normalised, matching the order used during evaluation.
            rms = rms_update(carry.rms, carry.obs.reshape(e * n, -1))
            obs_n = rms_normalize(rms, carry.obs)

            mean, log_std, new_hx = self.actor.apply(
                actor_params, obs_n.reshape(e * n, -1), carry.hx.reshape(e * n, -1)
            )
            std = jnp.exp(log_std)
            z = mean + std * jax.random.normal(step_key, mean.shape)
            action = jnp.tanh(z)
            log_prob = _tanh_normal_log_prob(z, mean, std, action).reshape(e, n)
            action = action.reshape(e, n, -1)

            value = self.critic.apply(critic_params, carry.gstate).squeeze(-1)

            env_state, next_obs, reward, term, done, info, next_gstate = self.env.step(
                carry.env_state, action
            )

            # Emitted as float32 masks: jax-metal 0.1.1 returns all-False for
            # bool-dtype scan outputs, which would drop every episode boundary.
            transition = Transition(
                obs=obs_n,
                gstate=carry.gstate,
                action=action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                term=term.astype(jnp.float32),
                done=done.astype(jnp.float32),
                hx=carry.hx,
                coverage=info['coverage_ratio'],
            )

            # Any episode end (collision or timeout) resets the recurrent state.
            # `done` here is the in-body value, unaffected by the scan-output bug.
            next_hx = new_hx.reshape(e, n, -1) * (1.0 - done.astype(jnp.float32))[:, None, None]
            return RolloutCarry(env_state, next_obs, next_gstate, next_hx, rms), transition

        carry, traj = jax.lax.scan(step, carry, jax.random.split(key, num_steps))
        last_value = self.critic.apply(critic_params, carry.gstate).squeeze(-1)
        return carry, traj, last_value

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------

    def _update(
        self,
        actor_state: TrainState,
        critic_state: TrainState,
        traj: Transition,
        advantages: jax.Array,
        returns: jax.Array,
        lr_actor: jax.Array,
        lr_critic: jax.Array,
    ) -> tuple[TrainState, TrainState, dict]:
        t, e, n = traj.obs.shape[0], traj.obs.shape[1], traj.obs.shape[2]
        flat = t * e * n

        # Each (t, e, i) triple is processed independently
        # (truncated-BPTT-length-1 approximation).
        obs_f = traj.obs.reshape(flat, -1)
        act_f = traj.action.reshape(flat, -1)
        old_log_prob = traj.log_prob.reshape(flat)
        hx_f = traj.hx.reshape(flat, -1)

        # Shared advantage: one value per (t, e), broadcast over N agents.
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_f = jnp.broadcast_to(adv[:, :, None], (t, e, n)).reshape(flat)

        states_f = traj.gstate.reshape(t * e, -1)
        returns_f = returns.reshape(t * e)

        def actor_loss_fn(params):
            mean, log_std, _ = self.actor.apply(params, obs_f, hx_f)
            std = jnp.exp(log_std)
            z = jnp.arctanh(jnp.clip(act_f, -1.0 + 1e-6, 1.0 - 1e-6))
            log_prob = _tanh_normal_log_prob(z, mean, std, act_f)

            ratio = jnp.exp(log_prob - old_log_prob)
            surr1 = ratio * adv_f
            surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_f
            loss = -jnp.mean(jnp.minimum(surr1, surr2))
            entropy = jnp.sum(_ENTROPY_CONST + log_std)
            return loss - self.entropy_coef * entropy, (loss, entropy)

        def critic_loss_fn(params):
            values = self.critic.apply(params, states_f).squeeze(-1)
            return 0.5 * jnp.mean((values - returns_f) ** 2)

        def epoch(carry, _):
            a_state, c_state = carry
            (_, (a_loss, entropy)), a_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(a_state.params)
            a_state = _apply_gradients(a_state, a_grads, lr_actor)

            c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(c_state.params)
            c_state = _apply_gradients(c_state, c_grads, lr_critic)
            return (a_state, c_state), (a_loss, c_loss, entropy)

        (actor_state, critic_state), (a_losses, c_losses, entropies) = jax.lax.scan(
            epoch, (actor_state, critic_state), None, length=self.n_epochs
        )
        metrics = {
            'actor_loss':  jnp.mean(a_losses),
            'critic_loss': jnp.mean(c_losses),
            'entropy':     jnp.mean(entropies),
        }
        return actor_state, critic_state, metrics

    def update(self, actor_state, critic_state, traj, advantages, returns,
               lr_actor, lr_critic):
        return self._update_fn(
            actor_state, critic_state, traj, advantages, returns,
            jnp.float32(lr_actor), jnp.float32(lr_critic),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def act_deterministic(
        self, actor_params, obs: jax.Array, hx: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Greedy action tanh(mean) for evaluation. obs/hx are (B, ...)."""
        mean, _, new_hx = self.actor.apply(actor_params, obs, hx)
        return jnp.tanh(mean), new_hx
