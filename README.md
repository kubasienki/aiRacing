# VDrift RL - Deep Reinforcement Learning for Racing

A complete reinforcement learning environment and training framework for autonomous racing in the [VDrift](http://vdrift.net) open-source racing simulator.

<p align="center">
  <span width="45%">
    
https://github.com/user-attachments/assets/f6be0b2d-ec22-4b56-b170-921a108a2bc0

  </span>
  <span width="45%">
    
https://github.com/user-attachments/assets/e093212f-bb0f-4e05-946e-22e6bc802174
    
  </span>
</p>
<p align="center">
  <a href="https://youtu.be/0yEg7_mNi3E">YouTube: Run 1</a> &nbsp;|&nbsp;
  <a href="https://youtu.be/nFH4dqV6qMs">YouTube: Run 2</a>
</p>

## Features

- **OpenAI Gym Environment**: Standard Gym interface for VDrift racing simulator
- **Vision-Based Control**: Train agents using 128x128 RGB camera observations
- **Realistic Physics**: Leverage VDrift's accurate vehicle dynamics simulation
- **Multi-Lap Racing**: Support for extended training episodes with lap counting
- **Flexible Configuration**: Configurable resolution, VDrift paths, and training parameters
- **Distributed Training**: Optional Redis-based support for multiple parallel environments
- **Pre-configured Examples**: PPO training scripts with TensorBoard logging

## Quick Start

```python
import gym
import vdrift_rl

# Create the environment
env = gym.make('VDrift-v0')

# Reset and run
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

## Installation

### Prerequisites

- Python 3.8+
- Linux (tested on Ubuntu 20.04+)
- Redis server (optional, for distributed training)
- Build tools for compiling VDrift

### Install Python Package

```bash
# Clone the repository
git clone --recursive https://github.com/kubasienki/aiRacing.git
cd aiRacing

# Install the package
pip install -e .
```

### Build Modified VDrift

See [docs/vdrift-setup.md](docs/vdrift-setup.md) for detailed instructions on building the modified VDrift simulator with RL support.

## Environment Details

### Observation Space

- **Type**: `Box(0, 255, (128, 128, 3), dtype=uint8)`
- **Description**: RGB image from driver's perspective
- **Configurable**: Resolution can be adjusted via constructor

### Action Space

- **Type**: `Box(5,)` with ranges:
  - `action[0]`: Throttle/Brake combined (-0.9 to 1.0)
  - `action[1]`: Clutch (0 to 1)
  - `action[2]`: Reserved
  - `action[3]`: Reserved
  - `action[4]`: Steering (-1 to 1, left to right)

### Reward Function

The reward function balances:
- **Progress**: Distance traveled along the track
- **Speed**: Velocity magnitude (with minimum speed requirements)
- **Track Position**: Penalties for going off-track
- **Collisions**: Negative reward for collisions
- **Lap Completion**: Bonus reward for completing laps

See [docs/environment.md](docs/environment.md) for detailed specifications.

## Training

### Basic Training Example

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import vdrift_rl

# Create vectorized environment
env = make_vec_env(lambda: gym.make('VDrift-v0'), n_envs=4)

# Train PPO agent
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=1000000)
model.save("vdrift_agent")
```

See [examples/](examples/) directory for complete training scripts with:
- Frame stacking
- Observation normalization
- Custom callbacks for logging
- Evaluation and video recording

## Project Structure

```
vdrift-rl/
├── src/vdrift_rl/          # Main Python package
│   ├── __init__.py
│   └── vdriftenv.py         # Gym environment implementation
├── vdrift/                  # Modified VDrift (git submodule)
├── examples/                # Training and evaluation scripts
├── tools/                   # Utility scripts (telemetry, client)
├── docs/                    # Documentation
├── archived_training/       # Archived models and training data
└── setup.py                 # Package installation
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Environment API](docs/environment.md)
- [VDrift Setup](docs/vdrift-setup.md)
- [Training Guide](docs/training.md) *(coming soon)*
- [Architecture Overview](docs/architecture.md) *(coming soon)*

## Research & Results

This project demonstrates successful training of vision-based racing agents using PPO.

<p align="center">
  <img src="docs/rotate.gif" width="60%" alt="Track overview"/>
</p>

Archived training runs include experiments with:
- State-dependent exploration (SDE)
- Collision penalty tuning
- Multi-lap training episodes
- Various reward function configurations

Training results and model checkpoints are available in `archived_training/`.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

- **Python Package** (`vdrift_rl`): MIT License - see [LICENSE](LICENSE)
- **Modified VDrift**: GNU General Public License v3 - see [vdrift/LICENSE](vdrift/LICENSE)

See [NOTICE](NOTICE) for full attribution and licensing details.

## Acknowledgments

- [VDrift Team](http://vdrift.net) for the excellent open-source racing simulator
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL implementations
- [OpenAI Gym](https://gym.openai.com/) for the environment interface

## Citation

If you use this project in your research, please cite:

```bibtex
@software{vdrift_rl,
  title = {VDrift RL: Deep Reinforcement Learning for Racing},
  author = {Jakub Sienkiewicz},
  year = {2025},
  url = {https://github.com/kubasienki/aiRacing}
}
```
