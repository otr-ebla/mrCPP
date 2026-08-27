import re

with open('src/envs/vec_env.py', 'r') as f:
    content = f.read()

init_old = "self.update_bosco = jax.jit(self._update_bosco)"
init_new = "self.update_bosco = jax.jit(self._update_bosco)\n        self.update_human_stop_prob = jax.jit(self._update_human_stop_prob)"
content = content.replace(init_old, init_new)

func_old = """    def _update_bosco(self, state: EnvState, targets: jax.Array):"""
func_new = """    def _update_human_stop_prob(self, state: EnvState, probs: jax.Array):
        def one(s: EnvState, p: jax.Array):
            return self.env.set_human_stop_prob(s, p)
        return jax.vmap(one)(state, probs)

    def _update_bosco(self, state: EnvState, targets: jax.Array):"""
content = content.replace(func_old, func_new)

with open('src/envs/vec_env.py', 'w') as f:
    f.write(content)

