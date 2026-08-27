import re

with open('src/train_simple.py', 'r') as f:
    content = f.read()

# Add human_stop_prob calculation in the loop
old_rollout = """        # ----------------------------------------------------------------
        # Collect T environment steps across all E envs in a single call
        # ----------------------------------------------------------------
        key, rollout_key = jax.random.split(key)
        prev_rms = carry.rms
        carry, traj, last_value = mappo.rollout(
            actor_state.params, critic_state.params, carry, T, rollout_key
        )"""

new_rollout = """        # ----------------------------------------------------------------
        # Collect T environment steps across all E envs in a single call
        # ----------------------------------------------------------------
        
        # Calculate human collision stop probability
        if update == start_update:
            smoothed_col_rate = 0.1
        else:
            total_col_rate = float(traj.wall_hit.mean()) + float(traj.robot_hit.mean())
            smoothed_col_rate = 0.9 * smoothed_col_rate + 0.1 * total_col_rate
            
        human_prob = max(0.0, min(1.0, 1.0 - (smoothed_col_rate / 0.1)))
        
        # update in env state
        carry = carry._replace(state=vec_env.update_human_stop_prob(carry.state, jnp.full((E,), human_prob, dtype=jnp.float32)))

        key, rollout_key = jax.random.split(key)
        prev_rms = carry.rms
        carry, traj, last_value = mappo.rollout(
            actor_state.params, critic_state.params, carry, T, rollout_key
        )"""

content = content.replace(old_rollout, new_rollout)

with open('src/train_simple.py', 'w') as f:
    f.write(content)

