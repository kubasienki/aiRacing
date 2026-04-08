import gym
from vdriftenv import VDriftEnv
import gym.wrappers.rescale_action as rescaler
import gym.wrappers.frame_stack as fstack
import gym.wrappers.normalize as norm

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import TensorBoardOutputFormat
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.logger import Image


import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback




class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.max_dist = 0.0
        self.deaths = 0
        self.max_dist_hist = []
        self.dist_hist = []
        self.end_type_hist = []
        self.end_points = []
        self.end_points_jump = []
        self.end_track_points = []
        self.end_track_points_jump = []

    def _log_episode_end(self, infos):
        for info in infos:
            if (info["ended"]):
                self.deaths += 1
                self.max_dist_hist.append(info["max_dist"])
                print(info["distance"])
                self.dist_hist.append(info["distance"])
                if info["too_slow"]:
                    self.end_type_hist.append(0)
                if info["colided"]:
                    self.end_type_hist.append(1)
                if info["out_of_track"]:
                    self.end_type_hist.append(2)
                if info["too_big_jump"]:
                    self.end_type_hist.append(3)
                    self.end_points_jump.append(info["position"])
                    self.end_track_points_jump.append(info["track_pos"])
                self.end_points.append(info["position"])
                self.end_track_points.append(info["track_pos"])
                self.logger.record_mean("quality/mean_end_dist", self.max_dist)


    def _generate_points_plot(self):
        fig, axs = plt.subplots(1)
        plt.ylim(-500, 500)
        plt.xlim(-700, 700)
        fig.set_figwidth(12)
        fig.set_figheight(12)
        if len(self.end_points) > 0:
            x_y, = axs.plot(np.array(self.end_points)[:,0], np.array(self.end_points)[:,1], '4') #tri right
            x1_y1, = axs.plot(np.array(self.end_track_points)[:,0], np.array(self.end_track_points)[:,1], "+")

        if len(self.end_points_jump) > 0:
            x3_y3, = axs.plot(np.array(self.end_points_jump)[:,0], np.array(self.end_points_jump)[:,1], "3") #tri left
            x2_y2, = axs.plot(np.array(self.end_track_points_jump)[:,0], np.array(self.end_track_points_jump)[:,1], "x")


        # force draw
        fig.canvas.draw()
        # save buffer
        w, h = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

        # display your plot
        # plt.pause(0.01)
        return buffer

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.deaths = 0
        self.max_dist = 0.0
        self.max_dist_hist = []
        self.dist_hist = []
        self.end_type_hist = []
        self.end_points = []
        self.end_track_points = []
        self.end_points_jump = []
        self.end_track_points_jump = []

        pass

    def _on_training_start(self):
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        self._log_episode_end(self.locals["infos"])

        self.max_dist = max(self.locals["infos"][0]["max_dist"], self.max_dist)
        self.logger.record("quality/max_dist", self.max_dist)
        self.logger.record_mean("quality/mean_velocity", self.locals["infos"][0]["velocity"])
        self.logger.record_mean("out_of_center", self.locals["infos"][0]["out_of_center"])

        return True

    def _on_rollout_end(self):
        # Log scalar value (here a random variable)
        self.logger.record("quality/deaths_per_rollout", self.deaths)
        self.logger.record("image/end_points", Image(self._generate_points_plot(), "HWC"),
                           exclude=("stdout", "log", "json", "csv"))
        if len(self.max_dist_hist) > 0:
            self.tb_formatter.writer.add_histogram("histogram/max_dist", np.array(self.max_dist_hist),
                                                   self.num_timesteps)
        if len(self.dist_hist) > 0:
            self.tb_formatter.writer.add_histogram("histogram/end_distance", np.array(self.dist_hist),
                                                   self.num_timesteps)
        if len(self.end_type_hist) > 0:
            self.tb_formatter.writer.add_histogram("histogram/end_reasons", np.array(self.end_type_hist),
                                                   self.num_timesteps)
            print("//////////////////")
        self.tb_formatter.writer.flush()


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
    env = VDriftEnv()
    # check_env(env)
    return env
# Parallel environments
def envFactoryEval():
    env = VDriftEnv()
    # check_env(env)
    return env


env = make_vec_env(envFactory, n_envs=6, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
env = VecNormalize.load("vdrift_normalization_stats_cnn_sde_off1", env)
env.training = True
env.norm_reward = True
env.norm_obs = False
env.clip_reward = 100.0
#env = VecNormalize(env, training=True, norm_obs=False, norm_reward=True, clip_reward=100.0)
env = VecFrameStack(env, 8)

# eval_env = envFactory()
# eval_env = VecFrameStack(env, 8)
# Use deterministic actions for evaluation
# eval_callback = EvalCallback(eval_env, best_model_save_path='./',
#                             log_path='./', eval_freq=250000,
#                             deterministic=False, render=False)




# Separate evaluation env
eval_env = make_vec_env(envFactoryEval, n_envs=1, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
eval_env = VecNormalize.load("vdrift_normalization_stats_cnn_sde_off1", eval_env)
eval_env.training = False
eval_env.norm_reward = True
eval_env.norm_obs = False
eval_env.clip_reward = 100.0
eval_env = VecFrameStack(eval_env, 8)
eval_env = VecTransposeImage(eval_env)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=100000,
                             deterministic=True, render=False)
callback = CallbackList([eval_callback, TensorboardCallback()])



#model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_cnn_tensorboard_5/", batch_size=512, n_epochs=3, learning_rate=0.0001) # use_sde=True, sde_sample_freq=4)
custom_objects = {
    "tensorboard_log": "./ppo_cnn_tensorboard_5/",
    "batch_size": 512,
    "n_epochs": 3,
    "learning_rate": 0.00001,
    #"use_sde": True,
    #"sde_sample_freq": 4
}
model = PPO.load("vdrift_model_cnn_sde_off1", env, verbose=1, custom_objects=custom_objects, tensorboard_log="./ppo_cnn_tensorboard_5/")  # = PPO("CnnPolicy", env, verbose=1)
# model = PPO.load("vdrift_model_cnn2")
for i in range(2, 10):
    model.learn(total_timesteps=15000000, callback=callback)
    model.save("vdrift_model_cnn_sde_off" + str(i))
    env.save("vdrift_normalization_stats_cnn_sde_off" + str(i))
    # env.render()

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    # print(obs)
    obs, rewards, dones, info = env.step(action)
