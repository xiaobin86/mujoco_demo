"""Run the Gymnasium API environment checker on PandaReachEnv."""

from gymnasium.utils.env_checker import check_env
from jaka_zu35_mujoco_rl.envs.panda_reach_env import PandaReachEnv


def main() -> None:
    env = PandaReachEnv()
    check_env(env, skip_render_check=True)
    print("check_env passed")


if __name__ == "__main__":
    main()
