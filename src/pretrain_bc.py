#!/usr/bin/env python3
"""
Behaviour cloning: pre-train the MAPPO actor on the BOSCO expert.

The expert drives the environment and the network learns the state -> action map
*while the rollout is running*: collection and gradient steps are interleaved in
chunks, so the dataset grows under the network instead of being frozen up front.

    obs (E, N, obs_dim)  --BOSCO-->  action (E, N, 2) in [-1, 1]
                         --actor-->  tanh(mean)

Only the actor is trained. The critic is initialised fresh and PPO learns it from
scratch; a warm actor with a cold critic is the usual and safe combination, since
the first PPO advantages are noise anyway and the clip keeps them from wrecking
the cloned policy in one update.

Why this works with the rest of the pipeline
--------------------------------------------
* Same observation pipeline. `train_simple` feeds the actor `rms_normalize(rms,
  obs)`, never the raw observation, so BC accumulates the very same
  `RunningMeanStd` over the expert's states and normalises with it during
  training. The statistics are saved in the checkpoint, so the policy that PPO
  resumes sees exactly the inputs it was cloned on.
* Same squashing. The actor's action is `tanh(mean)`; the loss is therefore
  applied to `tanh(mean)`, in the same [-1, 1] space the expert labels live in,
  not to the pre-squash mean (`atanh` is not even defined for the expert's
  frequent `v = 0`, which is exactly `a[0] = -1`).
* Same optimiser shape. The checkpoint is written with `train_simple`'s own
  `save_checkpoint`, with `update = 0` and a *freshly initialised* Adam state, so
  `python -m src.train_simple --resume <ckpt>` starts at update 1 with the full
  learning-rate schedule and no stale moments from the BC objective.

The exploration spread matters as much as the mean. PPO samples
`tanh(mean + sigma * eps)`, and at the actor's default sigma = 1 that is close to
bang-bang whatever the mean is — the first rollouts would throw away everything
BC just taught. `--init-std` (0.3 by default) therefore writes a narrower sigma
into the checkpoint; `entropy_coef` widens it again as PPO proceeds.

Model selection runs the policy, it does not read the loss
----------------------------------------------------------
Held-out MSE is computed on states the *expert* visits. The clone's own errors
take it off that distribution, and BOSCO's `v` is bang-bang — about half the
labels are exactly 0 (turning in place, yielding, recovering) and half are near
`v_max` — so a network that hedges between the two modes creeps instead of
driving, never tracks a lane, and compounds the error somewhere the loss is not
looking. Two checkpoints measured at 0.081 and 0.087 held-out MSE covered 10.5%
and 37.4% of the map. The loss is therefore a progress indicator only: every
`--eval-interval` chunks the live policy is rolled out greedily, and the
best-covering candidate is the one saved. Watch `cov` and `v` in the log — a
mean commanded `v` far below the expert's ~0.5 m/s is the creeping failure.

Covariate shift is the standard failure of behaviour cloning, and `--noise-std`
is the usual mitigation: Gaussian noise on the *executed* action with the stored
label left clean, so the expert demonstrates recovery from states just off its
own trajectory. It is off by default — on this expert it has not been shown to
help, and BOSCO reacts to a perturbation by stopping to re-aim, which feeds the
clone more of the v = 0 mode it is already prone to over-predicting.

More expert data is not the lever
--------------------------------
Measured on this task, longer cloning makes the closed-loop policy worse: over 64
episodes the loss fell 0.09 -> 0.02 while mean commanded v collapsed from 0.12 to
0.05 m/s and the best candidate reached only 19% coverage, against 37% from a
short, deliberately underfit 8-episode run whose v (0.43) still sat near the
label mean of 0.49. The v head cannot represent two modes, so the tighter it fits
the more of its mass lands on the absorbing one. Prefer a short run plus frequent
evaluation over a long run; `--episodes 8` to `--episodes 24` is the useful range
until the v channel is modelled properly (a mixture or a discrete speed head).

Usage
-----
    python -m src.pretrain_bc --episodes 8 --eval-interval 1
    python -m src.pretrain_bc --episodes 24 --save-data data/bc.npz
    python -m src.train_simple --resume checkpoints/bc_pretrained.pkl
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from src.algorithms.bosco import BoscoExpert
from src.algorithms.mappo import MAPPO, rms_init, rms_normalize, rms_update
from src.envs.vec_env import VecEnv
from src.models.actor_critic import _LOG_STD_MAX, _LOG_STD_MIN, Actor, Critic
from src.train_simple import save_checkpoint
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device

# atanh(+-1) is infinite, and the expert commands v = 0 (a[0] = -1) constantly,
# so the NLL target has to be pulled off the boundary.
_ATANH_CLIP = 1.0 - 1e-4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Buffer:
    """Flat (obs, action) store over (env, agent) samples, doubling on demand."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1 << 15):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.n = 0

    def _grow(self, need: int) -> None:
        cap = max(2 * self.obs.shape[0], need)
        for name in ('obs', 'act'):
            old = getattr(self, name)
            new = np.zeros((cap, old.shape[1]), np.float32)
            new[:self.n] = old[:self.n]
            setattr(self, name, new)

    def add(self, obs: np.ndarray, act: np.ndarray) -> None:
        k = obs.shape[0]
        if self.n + k > self.obs.shape[0]:
            self._grow(self.n + k)
        self.obs[self.n:self.n + k] = obs
        self.act[self.n:self.n + k] = act
        self.n += k

    def sample(self, rng: np.random.Generator, batch: int):
        idx = rng.integers(0, self.n, batch)
        return self.obs[idx], self.act[idx]

    def view(self):
        return self.obs[:self.n], self.act[:self.n]

    def recent(self, k: int) -> np.ndarray:
        """The last `k` observations actually written (never the spare capacity)."""
        return self.obs[max(0, self.n - k):self.n]


# ---------------------------------------------------------------------------
# Losses and the training step
# ---------------------------------------------------------------------------

def make_train_step(actor, loss_kind: str):
    """Jitted BC step. Normalisation happens *inside*, against the live `rms`."""

    def mse_loss(params, obs_n, target):
        mean, _ = actor.apply(params, obs_n)
        pred = jnp.tanh(mean)
        err = pred - target
        # Reported per action channel: v and omega have very different label
        # distributions (v is nearly bang-bang, omega is smooth), and a single
        # scalar hides which of the two is still wrong.
        return jnp.mean(err ** 2), jnp.mean(err ** 2, axis=0)

    def l1_loss(params, obs_n, target):
        """MAE on tanh(mean): fits the conditional MEDIAN rather than the mean.

        Kept for the record, and measured to be much worse than `mse` here. The
        idea was that the median of a bimodal label is one of its modes, so an L1
        fit would commit to stop-or-go rather than creep between them. It does
        commit — to *stop*: with about 47% of the v labels exactly 0 and the rest
        spread over the driving mode, zero wins the median nearly everywhere.
        Measured over 24 episodes: mean commanded v = 0.002 m/s and 0.78%
        coverage, i.e. robots that never leave their spawn cell.
        """
        mean, _ = actor.apply(params, obs_n)
        err = jnp.tanh(mean) - target
        return jnp.mean(jnp.abs(err)), jnp.mean(err ** 2, axis=0)

    def nll_loss(params, obs_n, target):
        """-log pi(a_expert | s) for the tanh-squashed Gaussian.

        The tanh Jacobian, -sum log(1 - a^2), depends only on the label, so it is
        dropped: what remains is a Gaussian NLL on z = atanh(a) that also fits
        `log_std` to the residual, leaving PPO a spread calibrated to how well
        the clone actually reproduces the expert.
        """
        mean, log_std = actor.apply(params, obs_n)
        z = jnp.arctanh(jnp.clip(target, -_ATANH_CLIP, _ATANH_CLIP))
        std = jnp.exp(log_std)
        nll = jnp.mean(0.5 * ((z - mean) / std) ** 2 + log_std)
        err = jnp.tanh(mean) - target
        return nll, jnp.mean(err ** 2, axis=0)

    loss_fn = {'mse': mse_loss, 'l1': l1_loss, 'nll': nll_loss}[loss_kind]

    @jax.jit
    def step(state: TrainState, rms, obs: jax.Array, act: jax.Array):
        def objective(params):
            return loss_fn(params, rms_normalize(rms, obs), act)

        (loss, per_dim), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), loss, per_dim

    return step


def make_eval_fn(actor, vec_env, steps: int, v_max: float):
    """Greedy on-policy rollout of the cloned actor: one episode per env.

    This, not the loss, is the quantity worth watching. Validation MSE is
    measured on states the *expert* visits; a clone that stalls immediately
    leaves that distribution and its error compounds where the loss cannot see
    it. Two checkpoints with the same held-out MSE have been observed to differ
    by 3.5x in coverage, so model selection has to run the policy.

    Returns (terminal coverage, mean commanded v). The second number is the
    diagnostic for the dominant failure mode: BOSCO's v is bang-bang (about half
    the labels are exactly 0, half near v_max), and a clone that hedges between
    the two modes creeps instead of driving, which reads here as a mean v far
    below the expert's ~0.5 * v_max.
    """

    @jax.jit
    def rollout(params, rms, key: jax.Array):
        state, obs, _, _ = vec_env.reset(key)

        def body(carry, _):
            state, obs = carry
            mean, _ = actor.apply(params, rms_normalize(rms, obs).reshape(-1, obs.shape[-1]))
            action = jnp.tanh(mean).reshape(obs.shape[0], obs.shape[1], -1)
            state, obs, _, _, done, info, _ = vec_env.step(state, action)
            # float32, not bool: jax-metal 0.1.1 returns all-False for bool-dtype
            # scan outputs, which would erase every episode boundary here.
            return (state, obs), (info['coverage_ratio'], done.astype(jnp.float32),
                                  jnp.mean(action[:, :, 0]))

        _, (cov, done, a0) = jax.lax.scan(body, (state, obs), None, length=steps)
        # Coverage sampled at the episode end (pre-reset, so it is the terminal
        # value); the fallback covers a horizon that ends mid-episode.
        finished = jnp.sum(done)
        coverage = jnp.where(finished > 0,
                             jnp.sum(cov * done) / jnp.maximum(finished, 1.0),
                             jnp.mean(cov[-1]))
        return coverage, (jnp.mean(a0) + 1.0) * 0.5 * v_max

    return rollout


def set_policy_std(params, std: float):
    """Rewrite the actor's state-independent sigma, inverting the sigmoid map.

    The default sigma = 1 is enormous next to a cloned action: `tanh(mean + z)`
    with z ~ N(0, 1) is close to bang-bang whatever the mean, so PPO's first
    rollouts would explore away everything BC just taught. Shrinking sigma here
    hands PPO a policy whose *sampled* behaviour, not just its mean, still looks
    like the expert.
    """
    t = (math.log(std) - _LOG_STD_MIN) / (_LOG_STD_MAX - _LOG_STD_MIN)
    if not 0.0 < t < 1.0:
        raise ValueError(
            f"--init-std {std} is outside the actor's representable range "
            f"({math.exp(_LOG_STD_MIN):.4f}, {math.exp(_LOG_STD_MAX):.4f})"
        )
    raw = math.log(t / (1.0 - t))
    inner = dict(params['params'])
    inner['log_std_raw'] = jnp.full_like(inner['log_std_raw'], raw)
    out = dict(params)
    out['params'] = inner
    return out


# ---------------------------------------------------------------------------
# Expert-driven collection
# ---------------------------------------------------------------------------

def make_experts(env, n: int) -> list[BoscoExpert]:
    """One expert per parallel env, sharing the (read-only) traversability graph.

    Building the graph costs a full sweep over every candidate edge of the map
    and is identical for all envs — they run the same layout. Every mutable piece
    of expert state (tours, cursors, counters) is created fresh in `reset`, so a
    shallow copy taken before the first reset never shares a writable array.
    """
    base = BoscoExpert(env)
    return [base] + [copy.copy(base) for _ in range(n - 1)]


def collect(vec_env, experts, state, obs, buffers, rng, steps, noise_std, stats):
    """Run `steps` expert steps on every env, filling the buffers in place.

    Returns the advanced `(state, obs)`. `buffers` is (train, val) — whole envs
    are held out, so validation measures generalisation to unseen spawns and
    unseen region partitions rather than memorisation of visited states.
    """
    train_buf, val_buf, n_val = buffers
    E, N = vec_env.E, vec_env.num_robots

    for _ in range(steps):
        pos = np.asarray(state.robot_positions)          # (E, N, 2)
        hdg = np.asarray(state.robot_headings)           # (E, N)
        cov = np.asarray(state.coverage_grid)            # (E, H, W)
        obs_np = np.asarray(obs)                         # (E, N, obs_dim)

        actions = np.stack([experts[e].act(pos[e], hdg[e], cov[e]) for e in range(E)])

        if n_val > 0:
            val_buf.add(obs_np[:n_val].reshape(-1, obs_np.shape[-1]),
                        actions[:n_val].reshape(-1, actions.shape[-1]))
        train_buf.add(obs_np[n_val:].reshape(-1, obs_np.shape[-1]),
                      actions[n_val:].reshape(-1, actions.shape[-1]))

        # The label is always the expert's clean action; only the executed one is
        # perturbed, which is what turns the noise into recovery supervision.
        executed = actions
        if noise_std > 0.0:
            executed = np.clip(
                actions + rng.normal(0.0, noise_std, actions.shape), -1.0, 1.0
            ).astype(np.float32)

        
        if step_idx == 0:
            smoothed_col_rate = 0.1
        else:
            total_col_rate = float(info['wall_collision_rate'].mean()) + float(info['robot_collision_rate'].mean())
            smoothed_col_rate = 0.9 * smoothed_col_rate + 0.1 * total_col_rate
            
        human_prob = max(0.0, min(1.0, 1.0 - (smoothed_col_rate / 0.1)))
        
        # update in env state
        state = vec_env.update_human_stop_prob(state, jnp.full((E,), human_prob, dtype=jnp.float32))

        state, obs, _, _, done, info, _ = vec_env.step(state, jnp.asarray(executed))

        stats['steps'] += E
        done_np = np.asarray(done)
        if done_np.any():
            coverage = np.asarray(info['coverage_ratio'])
            fresh_pos = np.asarray(state.robot_positions)   # already the new episode
            fresh_map = np.asarray(state.map_id)
            for e in np.flatnonzero(done_np):
                stats['episodes'] += 1
                stats['coverage'].append(float(coverage[e]))
                experts[e].reset(fresh_pos[e], map_id=int(fresh_map[e]))

    return state, obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pretrain(args) -> None:
    config = load_config(args.config)
    # Before any array exists, so implicit placement follows the chosen backend.
    device = select_device(None if args.backend == 'auto' else args.backend)
    print(f"Device: {describe(device)}  |  requested: {args.backend}")

    env_cfg = dict(config.get('env', {}))
    if args.max_steps is not None:
        env_cfg['max_steps'] = args.max_steps
    if args.humans > 0:
        env_cfg['num_humans'] = args.humans
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    vec_env = VecEnv(args.num_envs, env_cfg)
    env = vec_env.env
    E, N = vec_env.E, vec_env.num_robots
    # At least one env must stay in the training split.
    n_val = min(args.val_envs, E - 1) if E > 1 else 0

    actor = Actor(
        action_dim=vec_env.action_dim,
        vec_dim=env.obs_vec_dim,
        n_rays=env.n_rays,
        tail_dim=env.patch_dim,
        lidar_embed=model_cfg.get('lidar_embed',  64),
        hidden_size=model_cfg.get('hidden_size', 128),
    )
    critic = Critic(
        hidden_size=model_cfg.get('critic_hidden',    256),
        map_embed=model_cfg.get('critic_map_embed', 128),
    )
    # Only for the parameter initialisation (which runs on CPU when the backend
    # has no QR for the orthogonal init) and for the optimiser shapes PPO expects.
    mappo = MAPPO(actor, critic, vec_env, train_cfg, device=device)

    key = jax.random.PRNGKey(args.seed)
    key, init_key, reset_key, eval_key = jax.random.split(key, 4)
    actor_state, critic_state = mappo.create_train_states(init_key)

    bc_state = TrainState.create(
        apply_fn=actor.apply, params=actor_state.params, tx=optax.adam(args.lr)
    )
    rms = rms_init(vec_env.norm_dim)

    train_step = make_train_step(actor, args.loss)
    eval_fn = make_eval_fn(actor, vec_env, env.max_steps, env.v_max)
    # One fixed key for every selection eval, so the comparison across candidates
    # is on identical spawns; the final report uses a key never selected on.
    select_key, report_key = jax.random.split(eval_key)
    best = {'coverage': -1.0, 'params': None, 'where': 'none'}

    def evaluate(params, tag: str) -> str:
        """Score the live policy and keep it if it is the best seen so far."""
        cov, mean_v = eval_fn(params, rms, select_key)
        cov, mean_v = float(cov), float(mean_v)
        if cov > best['coverage']:
            best.update(coverage=cov, params=jax.device_get(params), where=tag)
        return f" | cov {cov:6.2%} | v {mean_v:.3f}"

    experts = make_experts(env, E)
    state, obs, _, _ = vec_env.reset(reset_key)
    map_ids = np.asarray(state.map_id)
    for e in range(E):
        experts[e].reset(np.asarray(state.robot_positions[e]),
                         map_id=int(map_ids[e]))

    train_buf = Buffer(vec_env.obs_dim, vec_env.action_dim)
    val_buf = Buffer(vec_env.obs_dim, vec_env.action_dim)
    rng = np.random.default_rng(args.seed)
    stats = {'steps': 0, 'episodes': 0, 'coverage': []}

    print(f"Envs: {E} ({n_val} held out for validation)  |  robots/env: {N}  |  "
          f"obs_dim: {vec_env.obs_dim}  |  horizon: {env.max_steps}")
    print(f"Map: {env.grid_h}x{env.grid_w} cells, {int(env.free_total)} coverable, "
          f"{int(experts[0]._isolated.sum())} unreachable by the robot disc")
    print(f"Loss: {args.loss}  |  lr: {args.lr}  |  batch: {args.batch}  |  "
          f"noise: {args.noise_std}")

    t0 = time.time()
    chunk = 0
    while stats['episodes'] < args.episodes:
        state, obs = collect(
            vec_env, experts, state, obs, (train_buf, val_buf, n_val),
            rng, args.chunk, args.noise_std, stats,
        )
        # Statistics see every observation exactly once, in the order it was
        # produced — the same running estimate PPO will keep extending.
        rms = rms_update(rms, jnp.asarray(train_buf.recent(args.chunk * (E - n_val) * N)))

        loss = per_dim = None
        for _ in range(args.grad_steps):
            o, a = train_buf.sample(rng, args.batch)
            bc_state, loss, per_dim = train_step(
                bc_state, rms, jnp.asarray(o), jnp.asarray(a)
            )
        chunk += 1

        evaluated = ''
        if args.eval_interval and chunk % args.eval_interval == 0:
            evaluated = evaluate(bc_state.params, f'chunk {chunk}')

        per_dim = np.asarray(per_dim)
        print(f"chunk {chunk:4d} | env_steps {stats['steps']:8d} | "
              f"samples {train_buf.n:8d} | episodes {stats['episodes']:4d} | "
              f"loss {float(loss):8.5f} | mse_v {per_dim[0]:.5f} | "
              f"mse_w {per_dim[1]:.5f}{evaluated} | {time.time() - t0:6.1f}s",
              flush=True)

    # A final set of epochs over the complete dataset: the early chunks were
    # fitted when the buffer held only the opening of an episode, so the network
    # has seen the end game far fewer times than the start.
    n_epochs_steps = max(1, train_buf.n // args.batch)
    for epoch in range(args.final_epochs):
        for _ in range(n_epochs_steps):
            o, a = train_buf.sample(rng, args.batch)
            bc_state, loss, per_dim = train_step(
                bc_state, rms, jnp.asarray(o), jnp.asarray(a)
            )
        val = ''
        if val_buf.n > 0:
            vo, va = val_buf.view()
            mean, _ = actor.apply(bc_state.params, rms_normalize(rms, jnp.asarray(vo)))
            val_mse = float(jnp.mean((jnp.tanh(mean) - jnp.asarray(va)) ** 2))
            val = f" | val_mse {val_mse:.5f}"
        # Every epoch is a selection candidate: the end of an epoch is an
        # arbitrary point on a loss surface that is flat in MSE and emphatically
        # not flat in closed-loop coverage.
        evaluated = evaluate(bc_state.params, f'epoch {epoch + 1}') if args.eval_interval else ''
        print(f"epoch {epoch + 1:3d}/{args.final_epochs} | loss {float(loss):8.5f} | "
              f"mse_v {float(per_dim[0]):.5f} | mse_w {float(per_dim[1]):.5f}{val}"
              f"{evaluated}", flush=True)

    expert_cov = float(np.mean(stats['coverage'])) if stats['coverage'] else float('nan')
    print(f"\nExpert: {stats['episodes']} episodes, "
          f"mean terminal coverage {expert_cov:.2%}")

    params = bc_state.params
    if args.eval_interval:
        # Held-out spawns: the selection key was used to *choose* this candidate,
        # so scoring on it again would report the maximum of a noisy sample.
        params = best['params']
        cov, mean_v = eval_fn(params, rms, report_key)
        print(f"Clone: selected {best['where']} (cov {best['coverage']:.2%} on the "
              f"selection key), {float(cov):.2%} on {E} unseen episodes, "
              f"mean commanded v {float(mean_v):.3f} m/s")

    if args.init_std is not None:
        params = set_policy_std(params, args.init_std)
        print(f"policy sigma set to {args.init_std} for the PPO phase")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # update = 0 so `--resume` restarts at update 1 with the full LR schedule, and
    # a fresh Adam state so PPO does not inherit moments fitted to the BC loss.
    save_checkpoint(
        args.out, 0,
        actor_state.replace(params=params, opt_state=actor_state.tx.init(params)),
        critic_state, rms,
    )
    print(f"saved pre-trained actor -> {args.out}  "
          f"({train_buf.n} training samples, {time.time() - t0:.1f}s)")
    print(f"continue with: python -m src.train_simple --resume {args.out}")

    if args.save_data:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_data)), exist_ok=True)
        o, a = train_buf.view()
        vo, va = val_buf.view()
        np.savez_compressed(args.save_data, obs=o, actions=a, val_obs=vo, val_actions=va)
        print(f"saved dataset -> {args.save_data}")


def main() -> None:
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser(
        description='Behaviour-cloning pre-training of the MAPPO actor on BOSCO')
    ap.add_argument('--config', default=os.path.join(here, '..', 'config',
                                                     'mappo_baseline.yaml'))
    ap.add_argument('--out', default=os.path.join(here, '..', 'checkpoints',
                                                  'bc_pretrained.pkl'))
    ap.add_argument('--episodes', type=int, default=24,
                    help='expert episodes to collect (across all envs)')
    ap.add_argument('--num-envs', type=int, default=8,
                    help='parallel envs; one BOSCO expert drives each')
    ap.add_argument('--val-envs', type=int, default=1,
                    help='envs held out for validation (0 disables)')
    ap.add_argument('--max-steps', type=int, default=None,
                    help='override the env horizon used for data collection')
    ap.add_argument('--chunk', type=int, default=200,
                    help='env steps collected between rounds of gradient steps')
    ap.add_argument('--grad-steps', type=int, default=200,
                    help='minibatch updates per chunk')
    ap.add_argument('--final-epochs', type=int, default=5,
                    help='epochs over the full dataset after collection ends')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--loss', default='mse', choices=['mse', 'l1', 'nll'],
                    help='mse: on tanh(mean), the best measured so far. '
                         'l1: fits the median, which collapses to v = 0 here. '
                         'nll: mse-like, but also fits log_std')
    ap.add_argument('--noise-std', type=float, default=0.0,
                    help='DART-style noise on the executed action; labels stay clean')
    ap.add_argument('--init-std', type=float, default=0.3,
                    help='policy sigma written into the checkpoint for the PPO '
                         'phase; the actor default of 1.0 samples away the clone')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-interval', type=int, default=5,
                    help='chunks between greedy evaluations; the best-scoring '
                         'candidate is what gets saved. 0 saves the last step '
                         'instead, which held-out MSE does not justify')
    ap.add_argument('--save-data', default=None,
                    help='also write the collected dataset to this .npz')
    ap.add_argument('--backend', default='auto',
                    choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'])
    ap.add_argument('--humans', nargs='?', type=int, const=3, default=0, help='Number of humans')
    pretrain(ap.parse_args())


if __name__ == '__main__':
    main()
