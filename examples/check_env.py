"""Run the Gymnasium API environment checker on PandaPickEnv."""

from gymnasium.utils.env_checker import check_env
from jaka_zu35_mujoco_rl import PandaPickEnv


def main() -> None:
    env = PandaPickEnv()
    check_env(env, skip_render_check=True)
    print("check_env passed")


if __name__ == "__main__":
    main()
