import re

with open('src/train_bosco.py', 'r') as f:
    content = f.read()

old_rollout = """        key, rollout_key = jax.random.split(key)
        prev_rms = carry.rms
        carry, traj, last_value, hits = rollout.run(
            actor_state.params, critic_state.params, carry, T, rollout_key
        )"""

new_rollout = """        if update == start_update:
            smoothed_col_rate = 0.1
        else:
            total_col_rate = float(traj.wall_hit.mean()) + float(traj.robot_hit.mean())
            smoothed_col_rate = 0.9 * smoothed_col_rate + 0.1 * total_col_rate
            
        human_prob = max(0.0, min(1.0, 1.0 - (smoothed_col_rate / 0.1)))
        
        # update in env state
        import jax.numpy as jnp
        carry = carry._replace(env_state=vec_env.update_human_stop_prob(carry.env_state, jnp.full((E,), human_prob, dtype=jnp.float32)))

        key, rollout_key = jax.random.split(key)
        prev_rms = carry.rms
        carry, traj, last_value, hits = rollout.run(
            actor_state.params, critic_state.params, carry, T, rollout_key
        )"""

content = content.replace(old_rollout, new_rollout)

with open('src/train_bosco.py', 'w') as f:
    f.write(content)

