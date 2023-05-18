import numpy as np
from matplotlib import pyplot as plt
import math
from vdriftenv import VDriftEnv

env = VDriftEnv()
empty_action = [0.0, 0.0, 0.0, 0.0, 0.0]


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
time_maxDistance, = axs[1].plot(time_x, maxDistance, '-')
time_reward, = axs[1].plot(time_x, rewards, '-')



def updateGraph():
    x_y.set_data(x, y)
    x1_y1.set_data(x1, y1)
    time_maxDistance.set_data(time_x, maxDistance)
    time_reward.set_data(time_x, rewards)
    axs[0].relim()
    axs[1].relim()
    axs[0].autoscale_view()
    axs[1].autoscale_view()
    plt.pause(0.01)
    return x_y,


env.reset()
observation, reward, ended, info = env.step(empty_action)
i = 0
while True:
    i += 1
    x.append(info["position"][0])
    y.append(info["position"][1])
    x1.append(info["track_pos"][0])
    y1.append(info["track_pos"][1])
    time_x.append(i * 0.05)
    maxDistance.append(reward)
    rewards.append(info["moving_forward"])
    updateGraph()
    observation, reward, ended, info = env.step(empty_action)
