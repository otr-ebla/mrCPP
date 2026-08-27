import re

with open('src/test_visual.py', 'r') as f:
    content = f.read()

old_mappo = """class MappoController:
    \"\"\"Trained policy. One jitted call per frame: normalise → act → step.

    Fusing the policy and the environment transition keeps a single device
    round-trip per rendered frame; only the render snapshot comes back.
    \"\"\"

    label = 'MAPPO '
    owner = None          # no partition to show"""

new_mappo = """class MappoController:
    \"\"\"Trained policy. One jitted call per frame: normalise → act → step.

    Fusing the policy and the environment transition keeps a single device
    round-trip per rendered frame; only the render snapshot comes back.
    \"\"\"

    label = 'MAPPO '
    owner = None

    def __init__(self, env: MultiRobotCoverageEnv, actor: Actor, params,
                 obs_rms: RunningMeanStd | None):
        self.env, self.params, self.obs_rms = env, params, obs_rms
        from src.algorithms.bosco import BoscoExpert
        self.expert = BoscoExpert(env)

        @jax.jit
        def policy_step(params, rms, state, obs):
            obs_n = rms_normalize(rms, obs) if rms is not None else obs
            mean, _ = actor.apply(params, obs_n)
            action = jnp.tanh(mean)                   # deterministic
            next_state, rewards, terminated, truncated = env.step(state, action)
            return next_state, env.get_obs(next_state), rewards, terminated, truncated

        self._fn = policy_step

    def reset(self, state) -> None:
        self.obs = self.env.get_obs(state)
        current_map_id = int(jax.device_get(state.map_id))
        self.expert.reset(np.asarray(state.robot_positions), map_id=current_map_id)
        owner = self.expert.owner.copy()
        owner[owner < 0] = self.env.num_robots
        self.owner = owner.reshape(self.env.grid_h, self.env.grid_w)"""

content = re.sub(
    r"class MappoController:.*?def reset\(self, state\) -> None:\n        self\.obs = self\.env\.get_obs\(state\)",
    new_mappo,
    content,
    flags=re.DOTALL
)

# Also we need to make sure `view_state['show_owner']` defaults to True for both?
# "Also during visualization of trained bosco policy show with light colors the colored cells as when visualizzing pure bosco algorithm."
content = content.replace("'show_owner': args.policy == 'bosco'", "'show_owner': True")

with open('src/test_visual.py', 'w') as f:
    f.write(content)

