import re

with open('src/pretrain_bc.py', 'r') as f:
    content = f.read()

old_rollout = """        state, obs, _, _, done, info, _ = vec_env.step(state, jnp.asarray(executed))"""
new_rollout = """        
        if step_idx == 0:
            smoothed_col_rate = 0.1
        else:
            total_col_rate = float(info['wall_collision_rate'].mean()) + float(info['robot_collision_rate'].mean())
            smoothed_col_rate = 0.9 * smoothed_col_rate + 0.1 * total_col_rate
            
        human_prob = max(0.0, min(1.0, 1.0 - (smoothed_col_rate / 0.1)))
        
        # update in env state
        state = vec_env.update_human_stop_prob(state, jnp.full((E,), human_prob, dtype=jnp.float32))

        state, obs, _, _, done, info, _ = vec_env.step(state, jnp.asarray(executed))"""

content = content.replace(old_rollout, new_rollout)

with open('src/pretrain_bc.py', 'w') as f:
    f.write(content)

