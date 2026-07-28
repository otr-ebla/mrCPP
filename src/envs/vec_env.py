"""
SubprocVecEnv: E independent MultiRobotCoverageEnv instances in separate processes.

Each worker owns one environment and communicates with the main process via a
multiprocessing Pipe.  The main process sends joint actions (N, 2) and receives
the full step result; environments that reach done auto-reset before replying.

step() returns both `terms` (hard collision only) and `dones` (term | trunc) so
callers can use the right signal for GAE masking vs hidden-state reset.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np


# ---------------------------------------------------------------------------
# Worker — runs in a subprocess, owns one environment
# ---------------------------------------------------------------------------

def _env_worker(conn: mp.connection.Connection, env_config: dict) -> None:
    """Subprocess entry point.  Must be a top-level function for spawn to pickle."""
    from src.envs.coverage_vector_env import MultiRobotCoverageEnv

    env = MultiRobotCoverageEnv(env_config)
    obs, info = env.reset()
    conn.send(('ready', obs, env.get_global_state(), info))

    while True:
        cmd, payload = conn.recv()
        if cmd == 'step':
            obs, rew, term, trunc, info = env.step(payload)
            state = env.get_global_state()
            done  = term or trunc
            if done:
                obs, info = env.reset()
                state = env.get_global_state()
            # Send term and trunc separately so the caller can use the correct
            # signal for GAE masking (term only) vs hidden-state reset (term|trunc).
            conn.send((obs, rew, term, trunc, info, state))
        elif cmd == 'reset':
            obs, info = env.reset()
            conn.send((obs, env.get_global_state(), info))
        elif cmd == 'close':
            env.close()
            conn.close()
            break


# ---------------------------------------------------------------------------
# SubprocVecEnv — runs in the main process
# ---------------------------------------------------------------------------

class SubprocVecEnv:
    """
    Vectorised wrapper that owns E environment processes.

    Environments are auto-reset on done: the returned observation and global
    state already belong to the new episode.  Hidden-state zeroing is the
    caller's responsibility (use the returned `dones` mask).
    """

    def __init__(self, num_envs: int, env_config: dict):
        self.E = num_envs
        ctx = mp.get_context('spawn')   # safe on all platforms

        self._conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_env_worker,
                args=(child_conn, env_config),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)

        init = [c.recv() for c in self._conns]   # ('ready', obs, state, info)
        self._obs_shape   = init[0][1].shape      # (N, obs_dim)
        self._state_shape = init[0][2].shape      # (state_dim,)

        self._last_obs   = np.stack([r[1] for r in init])
        self._last_state = np.stack([r[2] for r in init])
        self._last_info  = [r[3] for r in init]

    @property
    def num_robots(self) -> int:
        return self._obs_shape[0]

    @property
    def obs_dim(self) -> int:
        return self._obs_shape[1]

    @property
    def state_dim(self) -> int:
        return self._state_shape[0]

    def reset(self):
        """Reset all environments.  Returns (obs, states, infos)."""
        for conn in self._conns:
            conn.send(('reset', None))
        results = [c.recv() for c in self._conns]
        self._last_obs   = np.stack([r[0] for r in results])
        self._last_state = np.stack([r[1] for r in results])
        self._last_info  = [r[2] for r in results]
        return self._last_obs, self._last_state, self._last_info

    def step(self, actions: np.ndarray):
        """
        actions : (E, N, 2) in [-1, 1]

        Returns
        -------
        obs    : (E, N, obs_dim)  — from new episode when done
        rewards: (E, N)
        terms  : (E,) bool        — True on hard termination (collision only)
        dones  : (E,) bool        — True on terminated OR truncated
        infos  : list[dict]
        states : (E, state_dim)   — from new episode when done

        Use `terms` for GAE bootstrap masking (collision = no bootstrap).
        Use `dones` for hidden-state reset (any episode end resets the GRU).
        """
        for i, conn in enumerate(self._conns):
            conn.send(('step', actions[i]))
        results = [c.recv() for c in self._conns]   # (obs, rew, term, trunc, info, state)

        obs    = np.stack([r[0] for r in results])         # (E, N, obs_dim)
        rews   = np.stack([r[1] for r in results])         # (E, N)
        terms  = np.array([r[2] for r in results])         # (E,) bool — collision term
        truncs = np.array([r[3] for r in results])         # (E,) bool — timeout trunc
        infos  = [r[4] for r in results]
        states = np.stack([r[5] for r in results])         # (E, state_dim)
        dones  = terms | truncs                             # (E,) bool

        self._last_obs   = obs
        self._last_state = states
        self._last_info  = infos
        return obs, rews, terms, dones, infos, states

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(('close', None))
            except BrokenPipeError:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
