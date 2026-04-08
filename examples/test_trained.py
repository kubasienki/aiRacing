import vdrift_rl
import gym.wrappers.normalize as norm

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize

import gymnasium as gym


class NormObs(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, observation):
        """Clips the action within the valid bounds.
        Args:
            action: The action to clip
        Returns:
            The clipped action
        """
        return (observation - self.observation_space.low) / (
                self.observation_space.high + abs(self.observation_space.low))


# Parallel environments
def envFactory():
    env = gym.make('Vdrift-v0')
    # check_env(env)
    env = norm.NormalizeReward(env)
    return env

# env = VecNormalize(env, norm_obs=False)

# eval_env = envFactory()
# eval_env = VecFrameStack(env, 8)
# Use deterministic actions for evaluation
# eval_callback = EvalCallback(eval_env, best_model_save_path='./',
#                             log_path='./', eval_freq=250000,
#                             deterministic=False, render=False)




# Separate evaluation env
eval_env = make_vec_env(envFactory, n_envs=1)
eval_env = VecFrameStack(eval_env, 8)
# Use deterministic actions for evaluation


custom_objects = {
    "tensorboard_log": "./ppo_cnn_tensorboard_5/",
    "batch_size": 512,
    "n_epochs": 3,
    'observation_space': eval_env.observation_space,
    'action_space': eval_env.action_space
    }
model = PPO.load("./logs/best_model", eval_env, verbose=1, custom_objects=custom_objects, tensorboard_log="./ppo_cnn_tensorboard_5/")  # = PPO("CnnPolicy", env, verbose=1)
obs = eval_env.reset()
while True:
    action, _states = model.predict(obs)
    # print(obs)
    obs, rewards, dones, info = eval_env.step(action)
    print(info)
