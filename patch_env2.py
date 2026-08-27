import re

with open('src/envs/coverage_vector_env.py', 'r') as f:
    content = f.read()

old_term = """        human_terminated = jnp.any(human_collision) & terminate_on_human
        
        collided = (wall_hit | robot_hit | robot_hit_human) & alive
        alive_next = alive & ~collided if self.terminate_on_collision else (alive & ~human_collision if terminate_on_human else alive)

        dist_to_bosco_prev = jnp.sqrt(jnp.sum((state.robot_positions - state.bosco_targets)**2, axis=-1))
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

new_term = """        human_terminated = jnp.any(human_collision) & (terminate_on_human | self.terminate_on_collision)
        
        collided = (wall_hit | robot_hit | robot_hit_human) & alive
        alive_next = alive & ~collided if self.terminate_on_collision else (alive & ~human_collision if terminate_on_human else alive)

        dist_to_bosco_prev = jnp.sqrt(jnp.sum((state.robot_positions - state.bosco_targets)**2, axis=-1))
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

content = content.replace(old_term, new_term)

with open('src/envs/coverage_vector_env.py', 'w') as f:
    f.write(content)

