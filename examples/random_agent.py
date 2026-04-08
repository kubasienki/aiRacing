"""
Simple random agent example for VDrift RL environment.

This script demonstrates basic usage of the VDriftEnv environment
by running a random agent that takes random actions.

Usage:
    python random_agent.py
"""

import gym
import numpy as np

# Import the vdrift_rl package to register the environment
import vdrift_rl


def main():
    # Create the environment
    # Note: Make sure VDrift is built and the path is correct
    env = gym.make('VDrift-v0', use_redis=False)

    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run for 5 episodes
    num_episodes = 5

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

        while not (done or truncated):
            # Take random action
            action = env.action_space.sample()

            # Step the environment
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1

            # Print info every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: "
                      f"Reward: {reward:.2f}, "
                      f"Distance: {info['distance']:.1f}m, "
                      f"Speed: {info['velocity']:.1f} m/s")

            if done or truncated:
                print(f"\nEpisode finished after {step_count} steps")
                print(f"Total reward: {episode_reward:.2f}")
                print(f"Max distance: {info['max_dist']:.1f}m")
                print(f"Reason: ", end="")
                if info.get('ended'):
                    print("Timeout/slow/collision")
                elif info.get('too_big_jump'):
                    print("Teleportation detected")
                else:
                    print("Unknown")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
