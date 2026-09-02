from pathlib import Path

import pytest

from src.envs.coverage_vector_env import MultiRobotCoverageEnv
from src.utils.config_parser import load_config


CONFIG = Path(__file__).parent / "config" / "mappo_baseline.yaml"


def test_baseline_reward_weights_are_loaded_by_the_environment():
    config = load_config(CONFIG)
    env_config = config["env"]

    assert env_config["alpha"] == 10.0
    assert env_config["beta"] == 0.5
    assert env_config["kappa"] == 5.0
    assert env_config["tau"] == 0.05
    assert env_config["psi"] == 2.0
    assert env_config["bosco_gamma"] == 5.0
    assert "bosco_gamma" not in config["wandb"]


def test_redundancy_may_be_penalised_more_than_elapsed_time():
    env = MultiRobotCoverageEnv({"beta": 0.5, "tau": 0.05, "num_maps": 1})

    assert env.beta > env.tau


def test_negative_reward_weight_is_rejected():
    with pytest.raises(ValueError, match="reward weights must be non-negative: beta"):
        MultiRobotCoverageEnv({"beta": -0.1, "num_maps": 1})
