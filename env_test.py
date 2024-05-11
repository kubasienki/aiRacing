import numpy as np
from matplotlib import pyplot as plt
import math
from vdriftenv import VDriftEnv
import gym.wrappers.normalize as norm

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecVideoRecorder

import imageio
import numpy as np

empty_action = [-20.5, 0.0, 0.0, 0.0, 0.0]

max_images = 2000

x = []
x1 = []
y = []
y1 = []
time_x = []
maxDistance = []
rewards = []

fig, axs = plt.subplots(2)
x_y, = axs[0].plot(x, y, '-')
x1_y1, = axs[0].plot(x1, y1,"*")
time_maxDistance, = axs[1].plot(time_x, maxDistance, '--')
time_reward, = axs[1].plot(time_x, rewards, '-')






def envFactory():
    env = VDriftEnv(render_mode="rgb_array", fromStartLine=True)
    # check_env(env)
    env = norm.NormalizeReward(env)
    return env


def updateGraph():
    x_y.set_data(x, y)
    x1_y1.set_data(x1, y1)
    time_maxDistance.set_data(time_x, maxDistance)
    time_reward.set_data(time_x, rewards)
    axs[0].relim()
    axs[0].set_ylim([-0.1, 0.1])
    axs[1].relim()
    #axs[1].set_ylim([-0.1, 0.1])
    axs[0].autoscale_view()
    axs[1].autoscale_view()
    plt.pause(0.01)
    return x_y,


eval_env = make_vec_env(envFactory, n_envs=1)
eval_env = VecFrameStack(eval_env, 8)
eval_env = VecVideoRecorder(eval_env, "logs/video",
                           record_video_trigger=lambda x: x == 0, video_length=max_images,
                           name_prefix="ppo")
# Use deterministic actions for evaluation


custom_objects = {
    "tensorboard_log": "./ppo_cnn_tensorboard_5/",
    "batch_size": 512,
    "n_epochs": 3
}
model = PPO.load("./ppo_58", eval_env, verbose=1, custom_objects=custom_objects, tensorboard_log="./ppo_cnn_tensorboard_xxxx/")  # = PPO("CnnPolicy", env, verbose=1)

eval_env.reset()
observation = eval_env.reset()
i = 0
info = {}
images = []
gif = False
vid = False
rewardSum = 0.0
while True and not(i > max_images and (gif or vid)):
    i += 1
    action, _states = model.predict(observation)
    observation, reward, ended, info = eval_env.step(action)
    x.append(info[0]["position"][0])
    y.append(info[0]["position"][1])
    x1.append(info[0]["track_pos"][0])
    y1.append(info[0]["track_pos"][1])
    time_x.append(i * 0.05)
    maxDistance.append(reward[0])
    print(reward[0])
    rewardSum += reward[0]
    rewards.append(rewardSum)
    updateGraph()
    if gif:
        images.append(eval_env.render("rgb_array"))
    if ended:
        eval_env.reset()
        rewardSum = 0.0
if gif:
    imageio.mimsave("racing_ppo.gif", images, duration=50)
