import elements
import numpy as np

class EmbodiedEnvWrapper:
    """
    Wrapper che converte un ambiente custom/Gymnasium
    nell'interfaccia richiesta da DreamerV3 (elements/embodied).
    """
    def __init__(self, env):
        self._env = env
        self._done = True

    @property
    def obs_space(self):
        # Maps the environment's observation_space.
        base_space = self._env.observation_space
        obs_shape = base_space.shape
        obs_dtype = base_space.dtype

        return {
            'vector': elements.Space(obs_dtype, obs_shape),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        # Mappa l'action_space del tuo ambiente
        base_space = self._env.action_space
        return {
            'action': elements.Space(base_space.dtype, base_space.shape),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        # Gestione del Reset guidata dall'agente o dal fine episodio
        if action.get('reset', False) or self._done:
            self._done = False
            obs, _ = self._env.reset()
            return self._make_obs(obs, reward=0.0, is_first=True, is_last=False, is_terminal=False)

        # Step standard dell'ambiente
        obs, reward, terminated, truncated, info = self._env.step(action['action'])
        self._done = terminated or truncated

        # The env returns a per-robot reward vector; DreamerV3 treats the
        # whole multi-robot team as one centralized agent, so sum to a scalar.
        team_reward = np.asarray(reward).sum()

        return self._make_obs(
            obs=obs,
            reward=team_reward,
            is_first=False,
            is_last=self._done,
            is_terminal=terminated
        )

    def _make_obs(self, obs, reward, is_first, is_last, is_terminal):
        return {
            'vector': np.asarray(obs),
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

    def close(self):
        self._env.close()