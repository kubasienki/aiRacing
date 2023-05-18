# fmt: off
"""
Make your own custom environment
================================

This documentation overviews creating new environments and relevant
useful wrappers, utilities and tests included in Gymnasium designed for
the creation of new environments. You can clone gym-examples to play
with the code that is presented here. We recommend that you use a virtual environment:

.. code:: console

   git clone https://github.com/Farama-Foundation/gym-examples
   cd gym-examples
   python -m venv .env
   source .env/bin/activate
   pip install -e .

Subclassing gymnasium.Env
-------------------------

Before learning how to create your own environment you should check out
`the documentation of Gymnasium’s API </api/env>`__.

We will be concerned with a subset of gym-examples that looks like this:

.. code:: sh

   gym-examples/
     README.md
     setup.py
     gym_examples/
       __init__.py
       envs/
         __init__.py
         grid_world.py
       wrappers/
         __init__.py
         relative_position.py
         reacher_weighted_reward.py
         discrete_action.py
         clip_reward.py

To illustrate the process of subclassing ``gymnasium.Env``, we will
implement a very simplistic game, called ``GridWorldEnv``. We will write
the code for our custom environment in
``gym-examples/gym_examples/envs/grid_world.py``. The environment
consists of a 2-dimensional square grid of fixed size (specified via the
``size`` parameter during construction). The agent can move vertically
or horizontally between grid cells in each timestep. The goal of the
agent is to navigate to a target on the grid that has been placed
randomly at the beginning of the episode.

-  Observations provide the location of the target and agent.
-  There are 4 actions in our environment, corresponding to the
   movements “right”, “up”, “left”, and “down”.
-  A done signal is issued as soon as the agent has navigated to the
   grid cell where the target is located.
-  Rewards are binary and sparse, meaning that the immediate reward is
   always zero, unless the agent has reached the target, then it is 1.

An episode in this environment (with ``size=5``) might look like this:

where the blue dot is the agent and the red square represents the
target.

Let us look at the source code of ``GridWorldEnv`` piece by piece:
"""

# %%
# Declaration and Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our custom environment will inherit from the abstract class
# ``gymnasium.Env``. You shouldn’t forget to add the ``metadata``
# attribute to your class. There, you should specify the render-modes that
# are supported by your environment (e.g. ``"human"``, ``"rgb_array"``,
# ``"ansi"``) and the framerate at which your environment should be
# rendered. Every environment should support ``None`` as render-mode; you
# don’t need to add it in the metadata. In ``GridWorldEnv``, we will
# support the modes “rgb_array” and “human” and render at 4 FPS.
#
# The ``__init__`` method of our environment will accept the integer
# ``size``, that determines the size of the square grid. We will set up
# some variables for rendering and define ``self.observation_space`` and
# ``self.action_space``. In our case, observations should provide
# information about the location of the agent and target on the
# 2-dimensional grid. We will choose to represent observations in the form
# of dictionaries with keys ``"agent"`` and ``"target"``. An observation
# may look like ``{"agent": array([1, 0]), "target": array([0, 3])}``.
# Since we have 4 actions in our environment (“right”, “up”, “left”,
# “down”), we will use ``Discrete(4)`` as an action space. Here is the
# declaration of ``GridWorldEnv`` and the implementation of ``__init__``:

import numpy as np
import math

import gym
from gym import spaces
import socket
import struct

import matplotlib.pyplot as plt

import socket
import subprocess
import time
import random
import redis

H = 128
W = 128
photo_chunk_size = 4096 * 64

photo = True


class VDriftEnv(gym.Env):
    metadata = {"render_modes": ["text", "graphic"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        sock = socket.socket()
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        host = "localhost"

        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        url = r.lpop('vdrift')

        if url != None:
            address = url.split(":")
            port = address[1]
            host = address[0]
        else:
            subprocess.Popen(['./build/vdrift', '-resolution', str(H) + "x" + str(W), "-ai", str(port)], cwd="./vdrift")
            time.sleep(3)

        print("####################################server started     " + host + ":" + str(port))

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self._observation = np.zeros((H, W, 3), dtype='B')
        self._last_distance = 0.0
        self._max_distance = 0.0
        self._last_distance_from_mid = 0.0
        self._episode_start_distance = 0.0
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).

        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, 3), dtype='B')

        self.action_space = spaces.Box(low=np.array([-0.9, 0, -1, 0, -1]), high=np.array([1, 0, 1, 0, 1]), shape=(5,),
                                       dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # %%
    # Constructing Observations From Environment States
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Since we will need to compute observations both in ``reset`` and
    # ``step``, it is often convenient to have a (private) method ``_get_obs``
    # that translates the environment’s state into an observation. However,
    # this is not mandatory and you may as well compute observations in
    # ``reset`` and ``step`` separately:

    def _get_obs(self):
        return np.array(self._observation, dtype='B')

    # %%
    # We can also implement a similar method for the auxiliary information
    # that is returned by ``step`` and ``reset``. In our case, we would like
    # to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        return {}

    # %%
    # Oftentimes, info will also contain some data that is only available
    # inside the ``step`` method (e.g. individual reward terms). In that case,
    # we would have to update the dictionary that is returned by ``_get_info``
    # in ``step``.

    # %%
    # Reset
    # ~~~~~
    #
    # The ``reset`` method will be called to initiate a new episode. You may
    # assume that the ``step`` method will not be called before ``reset`` has
    # been called. Moreover, ``reset`` should be called whenever a done signal
    # has been issued. Users may pass the ``seed`` keyword to ``reset`` to
    # initialize any random number generator that is used by the environment
    # to a deterministic state. It is recommended to use the random number
    # generator ``self.np_random`` that is provided by the environment’s base
    # class, ``gymnasium.Env``. If you only use this RNG, you do not need to
    # worry much about seeding, *but you need to remember to call
    # ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
    # correctly seeds the RNG. Once this is done, we can randomly set the
    # state of our environment. In our case, we randomly choose the agent’s
    # location and the random sample target positions, until it does not
    # coincide with the agent’s position.
    #
    # The ``reset`` method should return a tuple of the initial observation
    # and some auxiliary information. We can use the methods ``_get_obs`` and
    # ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random

        resetPacket = struct.pack("ffffff??", 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, True, False)
        self.socket.sendall(resetPacket)
        data = self.socket.recv(1024)

        self._observation = np.zeros((H, W, 3), dtype='B')
        info = self._get_info()

        unpacked = struct.unpack("ffffffffffffffffffffffffffffff", data)

        # if self.render_mode == "graphic":
        #    self._render_frame()
        self._accSpeed = 500.0
        self._out_too_long = 0
        self._last_distance_from_mid = math.sqrt(math.pow(unpacked[14]-unpacked[0], 2) + math.pow(unpacked[15]-unpacked[1], 2))
        self._last_distance = unpacked[12]
        self._max_distance = unpacked[12]
        self._episode_start_distance = unpacked[12]
        return self._observation

    # %%
    # Step
    # ~~~~
    #
    # The ``step`` method usually contains most of the logic of your
    # environment. It accepts an ``action``, computes the state of the
    # environment after applying that action and returns the 4-tuple
    # ``(observation, reward, done, info)``. Once the new state of the
    # environment has been computed, we can check whether it is a terminal
    # state and we set ``done`` accordingly. Since we are using sparse binary
    # rewards in ``GridWorldEnv``, computing ``reward`` is trivial once we
    # know ``done``. To gather ``observation`` and ``info``, we can again make
    # use of ``_get_obs`` and ``_get_info``:
    def step(self, action):
        ended = False
        timestep = 0.05
        actionPacket = struct.pack("ffffff??", timestep, max(action[0], 0.0), action[1], -min(action[0], 0.0),
                                   action[3],
                                   max(min(action[4], 1.0), -1.0), False, True)
        self.socket.sendall(actionPacket)

        data = self.socket.recv(1024)
        unpacked = struct.unpack("ffffffffffffffffffffffffffffff", data)
        image = np.array([])
        if photo:
            self.socket.sendall(bytes("OK", 'ascii'))
            data = self.socket.recv(1024)
            size_data = data.decode('ascii')
            size = int(size_data[5:])
            # print(size, "\n\n\n\n\n\n\n")
            self.socket.sendall(bytes("OK", 'ascii'))
            total_received = 0
            total_data = b""

            while total_received < size:
                data = self.socket.recv(min(photo_chunk_size, size - total_received))
                total_received += len(data)
                total_data += data
            image = np.frombuffer(total_data, dtype='B')
            image_dimension = int(math.sqrt(total_received / 3))
            image = image.reshape((image_dimension, image_dimension, 3))
            self._observation = image

        # We use `np.clip` to make sure we don't leave the grid
        # An episode is done iff the agent has reached the target
        distance_from_mid_track = math.sqrt(math.pow(unpacked[14]-unpacked[0], 2) + math.pow(unpacked[15]-unpacked[1], 2))

        distance = max(unpacked[12] - self._max_distance, 0.0)
        moving_forward = (unpacked[12] - self._last_distance)
        moved_out_of_mid = (self._last_distance_from_mid - distance_from_mid_track)

        colided = unpacked[29] > 0.1
        out_of_track = unpacked[13] > 0.5

        if distance > 1500.0 or abs(moving_forward) > 1500.0:
            distance = 0.0
            moving_forward = 0.0
            self._out_too_long = 99999
            ended = True
        else:
            self._max_distance = max(unpacked[12], self._max_distance)
            self._last_distance = unpacked[12]
            self._last_distance_from_mid = distance_from_mid_track

        vel_sum = abs(unpacked[8]) + abs(unpacked[9]) + abs(unpacked[10])
        vel_sum_timestep_discounted = vel_sum * timestep
        # print(distance)
        reward = (moving_forward + distance * 10.0 ) \
                 + (moved_out_of_mid) \
                 #+ (-10.0 * ( 1.0 - vel_sum_timestep_discounted * 10.0) if vel_sum < 0.1 else vel_sum_timestep_discounted )
        #- 0.1 \
                 #+ (-200.0 if colided else 0.0) \
                 #+ (-100.0 if self._out_too_long > 1999 else 0)  # + (-0.5 if out_of_track else 0.0)
        # reward = reward / 50.0
        # if out_of_track:
        #   print(unpacked[14:])

        observation = self._get_obs()
        info = {
            'max_dist': self._max_distance - self._episode_start_distance,
            'out_of_center': self._last_distance_from_mid,
            "velocity": math.sqrt(unpacked[8]**2 + unpacked[9]**2 + unpacked[10]**2),
            "moving_forward": moving_forward,
            "max_distance_gain": distance,
            "position": [unpacked[0], unpacked[1], unpacked[2]],
            "track_pos": [unpacked[14], unpacked[15], unpacked[16]]
        }
        colision_timeout_factor = 100 #was: 1000
        out_of_track_timeout_factor = 0 #was: 0.1
        self._out_too_long += (10 if vel_sum < 0.5 else -0.5) + ( colision_timeout_factor if colided else 0) + (
            out_of_track_timeout_factor if out_of_track else 0.0)
        self._out_too_long = max(self._out_too_long, 0.0)
        return observation, reward, self._out_too_long > 2000 or ended, info

    # %%
    # Rendering
    # ~~~~~~~~~
    #
    # Here, we are using PyGame for rendering. A similar approach to rendering
    # is used in many environments that are included with Gymnasium and you
    # can use it as a skeleton for your own environments:

    def render(self, mode):
        if self.render_mode == "graphic":
            return self._render_frame()

    def _render_frame(self):
        y = [0.0] + [self._observation[10 + x * 3] for x in range(0, 5)]
        x = [0.0] + [self._observation[11 + x * 3] for x in range(0, 5)]
        plt.scatter(x, y)
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    # %%
    # Close
    # ~~~~~
    #
    # The ``close`` method should close any open resources that were used by
    # the environment. In many cases, you don’t actually have to bother to
    # implement this method. However, in our example ``render_mode`` may be
    # ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        if self.window is not None:
            self.socket.shutdown()
            self.socket.close()

# %%
# In other environments ``close`` might also close files that were opened
# or release other resources. You shouldn’t interact with the environment
# after having called ``close``.

# %%
# Registering Envs
# ----------------
#
# In order for the custom environments to be detected by Gymnasium, they
# must be registered as follows. We will choose to put this code in
# ``gym-examples/gym_examples/__init__.py``.
#
# .. code:: python
#
#   from gymnasium.envs.registration import register
#
#   register(
#        id="gym_examples/GridWorld-v0",
#        entry_point="gym_examples.envs:GridWorldEnv",
#        max_episode_steps=300,
#   )

# %%
# The environment ID consists of three components, two of which are
# optional: an optional namespace (here: ``gym_examples``), a mandatory
# name (here: ``GridWorld``) and an optional but recommended version
# (here: v0). It might have also been registered as ``GridWorld-v0`` (the
# recommended approach), ``GridWorld`` or ``gym_examples/GridWorld``, and
# the appropriate ID should then be used during environment creation.
#
# The keyword argument ``max_episode_steps=300`` will ensure that
# GridWorld environments that are instantiated via ``gymnasium.make`` will
# be wrapped in a ``TimeLimit`` wrapper (see `the wrapper
# documentation </api/wrappers>`__ for more information). A done signal
# will then be produced if the agent has reached the target *or* 300 steps
# have been executed in the current episode. To distinguish truncation and
# termination, you can check ``info["TimeLimit.truncated"]``.
#
# Apart from ``id`` and ``entrypoint``, you may pass the following
# additional keyword arguments to ``register``:
#
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | Name                 | Type      | Default   | Description                                                                                                   |
# +======================+===========+===========+===============================================================================================================+
# | ``reward_threshold`` | ``float`` | ``None``  | The reward threshold before the task is  considered solved                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``nondeterministic`` | ``bool``  | ``False`` | Whether this environment is non-deterministic even after seeding                                              |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``max_episode_steps``| ``int``   | ``None``  | The maximum number of steps that an episode can consist of. If not ``None``, a ``TimeLimit`` wrapper is added |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``order_enforce``    | ``bool``  | ``True``  | Whether to wrap the environment in an  ``OrderEnforcing`` wrapper                                             |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``autoreset``        | ``bool``  | ``False`` | Whether to wrap the environment in an ``AutoResetWrapper``                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``kwargs``           | ``dict``  | ``{}``    | The default kwargs to pass to the environment class                                                           |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
#
# Most of these keywords (except for ``max_episode_steps``,
# ``order_enforce`` and ``kwargs``) do not alter the behavior of
# environment instances but merely provide some extra information about
# your environment. After registration, our custom ``GridWorldEnv``
# environment can be created with
# ``env = gymnasium.make('gym_examples/GridWorld-v0')``.
#
# ``gym-examples/gym_examples/envs/__init__.py`` should have:
#
# .. code:: python
#
#    from gym_examples.envs.grid_world import GridWorldEnv
#
# If your environment is not registered, you may optionally pass a module
# to import, that would register your environment before creating it like
# this - ``env = gymnasium.make('module:Env-v0')``, where ``module``
# contains the registration code. For the GridWorld env, the registration
# code is run by importing ``gym_examples`` so if it were not possible to
# import gym_examples explicitly, you could register while making by
# ``env = gymnasium.make('gym_examples:gym_examples/GridWorld-v0)``. This
# is especially useful when you’re allowed to pass only the environment ID
# into a third-party codebase (eg. learning library). This lets you
# register your environment without needing to edit the library’s source
# code.

# %%
# Creating a Package
# ------------------
#
# The last step is to structure our code as a Python package. This
# involves configuring ``gym-examples/setup.py``. A minimal example of how
# to do so is as follows:
#
# .. code:: python
#
#    from setuptools import setup
#
#    setup(
#        name="gym_examples",
#        version="0.0.1",
#        install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
#    )
#
# Creating Environment Instances
# ------------------------------
#
# After you have installed your package locally with
# ``pip install -e gym-examples``, you can create an instance of the
# environment via:
#
# .. code:: python
#
#    import gym_examples
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#
# You can also pass keyword arguments of your environment’s constructor to
# ``gymnasium.make`` to customize the environment. In our case, we could
# do:
#
# .. code:: python
#
#    env = gymnasium.make('gym_examples/GridWorld-v0', size=10)
#
# Sometimes, you may find it more convenient to skip registration and call
# the environment’s constructor yourself. Some may find this approach more
# pythonic and environments that are instantiated like this are also
# perfectly fine (but remember to add wrappers as well!).
#
# Using Wrappers
# --------------
#
# Oftentimes, we want to use different variants of a custom environment,
# or we want to modify the behavior of an environment that is provided by
# Gymnasium or some other party. Wrappers allow us to do this without
# changing the environment implementation or adding any boilerplate code.
# Check out the `wrapper documentation </api/wrappers/>`__ for details on
# how to use wrappers and instructions for implementing your own. In our
# example, observations cannot be used directly in learning code because
# they are dictionaries. However, we don’t actually need to touch our
# environment implementation to fix this! We can simply add a wrapper on
# top of environment instances to flatten observations into a single
# array:
#
# .. code:: python
#
#    import gym_examples
#    from gymnasium.wrappers import FlattenObservation
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = FlattenObservation(env)
#    print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
#
# Wrappers have the big advantage that they make environments highly
# modular. For instance, instead of flattening the observations from
# GridWorld, you might only want to look at the relative position of the
# target and the agent. In the section on
# `ObservationWrappers </api/wrappers/#observationwrapper>`__ we have
# implemented a wrapper that does this job. This wrapper is also available
# in gym-examples:
#
# .. code:: python
#
#    import gym_examples
#    from gym_examples.wrappers import RelativePosition
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = RelativePosition(env)
#    print(wrapped_env.reset())     # E.g.  [-3  3], {}
