import pathlib
from functools import partial as bind

import elements
import embodied
import numpy as np
import ruamel.yaml as yaml
from dreamerv3 import agent as agent_lib

from src.envs import make_custom_env
from src.dreamer_wrapper import EmbodiedEnvWrapper


def make_env(config, index):
    raw_env = make_custom_env()
    env = EmbodiedEnvWrapper(raw_env)
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if name != 'reset' and not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_agent(config):
    env = make_env(config, 0)
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    return agent_lib.Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


def make_replay(config, folder, mode='train'):
    batlen = config.batch_length if mode == 'train' else config.report_length
    consec = config.consec_train if mode == 'train' else config.consec_report
    capacity = config.replay.size if mode == 'train' else config.replay.size / 10
    length = consec * batlen + config.replay_context
    directory = elements.Path(config.logdir) / folder
    return embodied.replay.Replay(
        length=length,
        capacity=int(capacity),
        online=config.replay.online,
        chunksize=config.replay.chunksize,
        directory=directory,
    )


def make_stream(config, replay, mode):
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True,
    )
    return stream


def make_logger(config):
    step = elements.Counter()
    logdir = config.logdir
    outputs = [
        elements.logger.TerminalOutput(config.logger.filter, 'Agent'),
        elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        elements.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
    ]
    return elements.Logger(step, outputs)


def main():
    print("Starting DreamerV3 training on the custom environment...")

    config_path = pathlib.Path(agent_lib.__file__).parent / 'configs.yaml'
    with open(config_path, 'r') as f:
        configs = yaml.YAML(typ='safe').load(f)

    config = elements.Config(configs['defaults'])
    if 'size12m' in configs:
        config = config.update(configs['size12m'])

    config = config.update({
        'logdir': './logdir/custom_env_run',
        'jax.platform': 'cpu',  # or 'gpu' on Linux/CUDA
        'batch_size': 16,
        'batch_length': 64,
        'run.envs': 1,
        'run.debug': True,     # keep the env in-process (no multiprocessing)
        'run.steps': 10_000,
    })

    logdir = elements.Path(config.logdir)
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args,
    )


if __name__ == '__main__':
    main()
