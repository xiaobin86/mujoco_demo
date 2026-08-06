"""Run a random agent for a few episodes in PandaPickEnv."""

from jaka_zu35_mujoco_rl import PandaPickEnv


def main() -> None:
    env = PandaPickEnv()

    num_episodes = 3
    max_steps = 500

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        while not (terminated or truncated) and step < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

        print(
            f"Episode {episode + 1}: "
            f"steps={step}, reward={episode_reward:.4f}, "
            f"success={terminated}, cube_to_tray={info['distance_cube_to_tray']:.4f}"
        )

    env.close()


if __name__ == "__main__":
    main()
