"""Run the Gymnasium API environment checker on JakaReachEnv."""

from gymnasium.utils.env_checker import check_env
from jaka_zu35_mujoco_rl.envs.jaka_reach_env import JakaReachEnv


def main() -> None:
    env = JakaReachEnv()
    check_env(env, skip_render_check=True)
    print("check_env passed")


if __name__ == "__main__":
    main()
