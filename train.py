import gym
from vdriftenv import VDriftEnv
import gym.wrappers.rescale_action as rescaler
import gym.wrappers.frame_stack as fstack
import gym.wrappers.normalize as norm

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.max_dist = 0.0

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.max_dist = 0.0
        pass

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.max_dist = max(self.locals["infos"][0]["max_dist"], self.max_dist)
        self.logger.record("max_dist", self.max_dist)
        self.logger.record_mean("out_of_center", self.locals["infos"][0]["out_of_center"])

        return True

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
        return (observation - self.observation_space.low)/(self.observation_space.high + abs(self.observation_space.low))


# Parallel environments
def envFactory():
    env = VDriftEnv()
    # check_env(env)
    env = norm.NormalizeReward(env)
    return env

env = make_vec_env(envFactory, n_envs=6, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
env = VecFrameStack(env, 8)
#env = VecNormalize(env, norm_obs=False)

#eval_env = envFactory()
#eval_env = VecFrameStack(env, 8)
# Use deterministic actions for evaluation
#eval_callback = EvalCallback(eval_env, best_model_save_path='./',
#                             log_path='./', eval_freq=250000,
#                             deterministic=False, render=False)




#callback = CallbackList([eval_callback, TensorboardCallback()])

# PPO.load("vdrift_model", env, verbose=1)# = PPO("CnnPolicy", env, verbose=1)
#model = A2C("CnnPolicy", env, verbose=1, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_cnn_tensorboard_3/")
#model = PPO.load("vdrift_model_cnn2")
for i in range(2, 10):
    model.learn(total_timesteps=5000000, callback=TensorboardCallback())
    model.save("vdrift_model_cnn" + str(i))
    # env.render()

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    #print(obs)
    obs, rewards, dones, info = env.step(action)
