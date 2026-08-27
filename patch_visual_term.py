import re

with open('src/test_visual.py', 'r') as f:
    content = f.read()

old_cfg = """    if args.humans > 0:
        env_cfg['num_humans'] = args.humans"""
new_cfg = """    if args.humans > 0:
        env_cfg['num_humans'] = args.humans
    env_cfg['terminate_on_collision'] = True"""

content = content.replace(old_cfg, new_cfg)

with open('src/test_visual.py', 'w') as f:
    f.write(content)

