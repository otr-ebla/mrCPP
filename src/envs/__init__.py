from src.envs.coverage_vector_env import MultiRobotCoverageEnv


def make_custom_env(config: dict | None = None) -> MultiRobotCoverageEnv:
    """Factory for the multi-robot coverage environment."""
    return MultiRobotCoverageEnv(config)
