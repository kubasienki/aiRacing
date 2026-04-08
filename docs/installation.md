# Installation Guide

This guide walks you through setting up the VDrift RL environment for training reinforcement learning agents.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian, or similar)
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended for parallel training)
- **Disk**: 5GB free space
- **GPU**: Optional but recommended for faster training

### Required Software
- Python 3.8+
- Git
- Build tools (gcc, g++, make, cmake)
- Redis server (optional, for distributed training)

## Step-by-Step Installation

### 1. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    git build-essential cmake \
    redis-server \
    libsdl2-dev libsdl2-image-dev libsdl2-net-dev \
    libvorbis-dev libgl1-mesa-dev libglu1-mesa-dev \
    libarchive-dev libcurl4-openssl-dev
```

#### Arch Linux
```bash
sudo pacman -S python python-pip git base-devel cmake redis \
    sdl2 sdl2_image sdl2_net libvorbis mesa glu libarchive curl
```

### 2. Clone the Repository

```bash
git clone --recursive https://github.com/kubasienki/aiRacing.git
cd aiRacing
```

**Note**: The `--recursive` flag is important to clone the VDrift submodule.

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

### 3. Set Up Python Environment

Using virtualenv (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or using conda:
```bash
conda create -n vdrift-rl python=3.9
conda activate vdrift-rl
```

### 4. Install Python Package

Install in development mode (recommended for development):
```bash
pip install -e .
```

Or install normally:
```bash
pip install .
```

### 5. Install Training Dependencies (Optional)

For training with Stable-Baselines3:
```bash
pip install -r requirements-training.txt
```

For development tools:
```bash
pip install -r requirements-dev.txt
```

### 6. Build Modified VDrift

See [vdrift-setup.md](vdrift-setup.md) for detailed instructions on building VDrift.

Quick build:
```bash
cd vdrift
scons
cd ..
```

The VDrift executable will be at `vdrift/build/vdrift`.

### 7. Configure Environment Variables (Optional)

Set the path to your VDrift executable:
```bash
export VDRIFT_BIN=/path/to/vdrift-rl/vdrift/build/vdrift
```

Add to your `~/.bashrc` or `~/.zshrc` to make it permanent.

### 8. Verify Installation

Test the environment:
```python
import gym
import vdrift_rl

env = gym.make('VDrift-v0')
obs = env.reset()
print("Environment created successfully!")
print(f"Observation shape: {obs.shape}")
env.close()
```

Or run a random agent:
```bash
cd examples
python examples/vdriftenv_test.py
```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'vdrift_rl'

Make sure you installed the package:
```bash
pip install -e .
```

And that you're using the correct Python environment.

#### 2. VDrift executable not found

Set the `VDRIFT_BIN` environment variable:
```bash
export VDRIFT_BIN=/absolute/path/to/vdrift-rl/vdrift/build/vdrift
```

Or specify it when creating the environment:
```python
env = gym.make('VDrift-v0', vdrift_path='/path/to/vdrift')
```

#### 3. Redis connection failed

If not using distributed training, you can disable Redis:
```python
env = gym.make('VDrift-v0', use_redis=False)
```

Or start Redis server:
```bash
sudo systemctl start redis
# or
redis-server
```

#### 4. VDrift fails to start - SDL errors

Make sure you have X11 or Wayland display:
```bash
echo $DISPLAY
```

For headless servers, you may need to use Xvfb:
```bash
sudo apt install xvfb
xvfb-run python your_training_script.py
```

#### 5. Permission denied when running VDrift

Make the binary executable:
```bash
chmod +x vdrift/build/vdrift
```

## Next Steps

- Read the [Environment API documentation](environment.md)
- Check out [example training scripts](../examples/)
- Learn about [VDrift setup and modifications](vdrift-setup.md)

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Review the documentation in `docs/`
