import jax
import jax.numpy as jnp
from src.models.actor_critic import Actor

actor = Actor(
    action_dim=2,
    vec_dim=4,
    n_rays=70,
    tail_dim=4,
    lidar_embed=64,
    hidden_size=128
)
obs = jnp.zeros((1, 78))

with jax.default_device(jax.devices("cpu")[0]):
    params = actor.init(jax.random.PRNGKey(0), obs)
    print("init shape:", params['params']['Dense_1']['kernel'].shape)
