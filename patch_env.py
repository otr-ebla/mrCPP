import re

with open('src/envs/coverage_vector_env.py', 'r') as f:
    content = f.read()

# 1. Add bosco_gamma to init
content = content.replace(
    "self.alpha = float(cfg.get('alpha', 10.0))",
    "self.alpha = float(cfg.get('alpha', 10.0))\n        self.bosco_gamma = float(cfg.get('bosco_gamma', 1.0))"
)

# 2. Add human_stop_prob to EnvState
content = content.replace(
    "robot_hits:       jax.Array   # (N,)     float32  — 0.0 / 1.0",
    "robot_hits:       jax.Array   # (N,)     float32  — 0.0 / 1.0\n    human_stop_prob:  jax.Array   # ()       float32"
)

# 3. Add human_stop_prob to reset
content = content.replace(
    "robot_hits       = jnp.zeros((self.num_robots,), jnp.float32),",
    "robot_hits       = jnp.zeros((self.num_robots,), jnp.float32),\n            human_stop_prob  = jnp.float32(0.0),"
)

# 4. Modify step function for reward and human collision prob
old_step_code = """        key, h_hdg_key, h_dist_key = jax.random.split(state.key, 3)"""
new_step_code = """        key, h_hdg_key, h_dist_key, h_stop_key = jax.random.split(state.key, 4)"""
content = content.replace(old_step_code, new_step_code)

old_collision = """        collided = (wall_hit | robot_hit | robot_hit_human) & alive
        alive_next = alive & ~collided if self.terminate_on_collision else alive"""
new_collision = """        terminate_on_human = jax.random.uniform(h_stop_key) < state.human_stop_prob
        human_collision = robot_hit_human & alive
        human_terminated = jnp.any(human_collision) & terminate_on_human
        
        collided = (wall_hit | robot_hit | robot_hit_human) & alive
        alive_next = alive & ~collided if self.terminate_on_collision else (alive & ~human_collision if terminate_on_human else alive)"""
content = content.replace(old_collision, new_collision)

old_reward = """        rewards = jnp.where(
            alive,
            self.alpha * discovered
            - self.beta * redundant
            - self.tau
            - self.kappa * collided
            - prox_pen
            + team_bonus,
            0.0,
        ).astype(jnp.float32)

        step_count = state.step_count + 1
        truncated  = step_count >= self.max_steps
        terminated = complete | (jnp.any(collided) if self.terminate_on_collision
                                 else jnp.bool_(False))"""

new_reward = """        dist_to_bosco_prev = jnp.sqrt(jnp.sum((state.robot_positions - state.bosco_targets)**2, axis=-1))
        dist_to_bosco_next = jnp.sqrt(jnp.sum((new_pos - state.bosco_targets)**2, axis=-1))
        bosco_reward_term = self.bosco_gamma * (dist_to_bosco_prev - dist_to_bosco_next)

        rewards = jnp.where(
            alive,
            self.alpha * discovered
            - self.beta * redundant
            - self.tau
            - self.kappa * collided
            - prox_pen
            + team_bonus
            + bosco_reward_term,
            0.0,
        ).astype(jnp.float32)

        step_count = state.step_count + 1
        truncated  = step_count >= self.max_steps
        terminated = complete | human_terminated | (jnp.any(wall_hit | robot_hit) if self.terminate_on_collision else jnp.bool_(False))"""

content = content.replace(old_reward, new_reward)

# 5. Add set_human_stop_prob
add_func = """    def set_bosco_targets(self, state: EnvState, bosco_targets: jax.Array) -> EnvState:"""
new_func = """    def set_human_stop_prob(self, state: EnvState, prob: jax.Array) -> EnvState:
        return state.replace(human_stop_prob=prob)

    def set_bosco_targets(self, state: EnvState, bosco_targets: jax.Array) -> EnvState:"""
content = content.replace(add_func, new_func)

with open('src/envs/coverage_vector_env.py', 'w') as f:
    f.write(content)

