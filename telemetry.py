import socket
import struct
import time
import numpy as np
from matplotlib import pyplot as plt
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange

x = []
x1 = []
y = []
y1 = []
time_x = []
maxDistance = []
rewards = []
total_reward = 0.0
max_distance = 0.0
last_distance = 0.0

fig, axs = plt.subplots(2)
x_y, = axs[0].plot(x, y, '-')
x1_y1, = axs[0].plot(x1, y1, '*')
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


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("localhost", 8081))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
i = 1
sleep_factor = 0
photo_chunk_size = 4096 * 64
photo = True
total_time = 0.0
timestep = 0.05
actionPacket = struct.pack("ffffff??", timestep, 0.8, 0.0, 0.0, 0.0, 0.0, False, photo)
s.sendall(actionPacket)

out_too_long = 0
last_distance_from_mid_track = 0.0
while True:
    updateGraph()
    total_time += timestep
    start_time = time.time()
    data = s.recv(1024)
    unpacked = struct.unpack("ffffffffffffffffffffffffffffff", data)
    if (unpacked[14] != 0.0):
        x1.append(unpacked[14])
        y1.append(unpacked[15])
        x.append(unpacked[0])
        y.append(unpacked[1])
    else:
        x1.append(x1[-1])
        y1.append([y1[-1]])
        x.append(unpacked[0])
        y.append(unpacked[1])


    time_x.append(total_time)
    colided = unpacked[29] > 0.1
    out_of_track = unpacked[13] > 0.5
    distance_from_mid_track = math.sqrt(math.pow(unpacked[14]-unpacked[0], 2) + math.pow(unpacked[15]-unpacked[1], 2))


    distance_reported = unpacked[12]
    distance = max(distance_reported - max_distance, 0.0)
    moving_forward = (distance_reported - last_distance)

    if distance > 1500.0 or abs(moving_forward) > 1500.0:
        distance = 0.0
        moving_forward = 0.0
        out_too_long = 99999
        ended = True
    else:
        max_distance = max(distance_reported, max_distance)
        last_distance = distance_reported

    vel_sum = abs(unpacked[8]) + abs(unpacked[9]) + abs(unpacked[10])
    # print(distance)
    reward = moving_forward + distance * 10.0 - 0.1 + (
        -10.0 * (1.0 - vel_sum * 10.0) if vel_sum < 0.1 else vel_sum / 5.0) - (200.0 if colided else 0.0) + (
                 -100.0 if out_too_long > 1999 else 0) + (-0.5 if out_of_track else 0.0)
    out_too_long = 0.0
    total_reward += reward
    rewards.append(last_distance_from_mid_track-distance_from_mid_track)
    last_distance_from_mid_track = distance_from_mid_track
    maxDistance.append(0.0)

    total_data = b""
    if photo:
        s.sendall(bytes("OK", 'ascii'))
        data = s.recv(1024)
        size_data = data.decode('ascii')
        size = int(size_data[5:])
        # print(size, "\n\n\n\n\n\n\n")
        s.sendall(bytes("OK", 'ascii'))
        total_received = 0

    actionPacket = struct.pack("ffffff??", timestep, 0.8, 0.0, 0.0, 0.0, 0.0, False, photo)
    s.sendall(actionPacket)
    if photo:
        while total_received < size:
            data = s.recv(min(photo_chunk_size, size - total_received))
            total_received += len(data)
            total_data += data
        data = np.frombuffer(total_data, dtype='B')
        image_dimension = int(math.sqrt(total_received / 3))
        data = data.reshape((image_dimension, image_dimension, 3))
        # plt.imshow(data, interpolation='nearest')
        # plt.show()

    time.sleep(sleep_factor)
