# VDrift Environment API Documentation

Complete API reference for the `VDriftEnv` Gym environment.

## Environment Registration

```python
import gym
import vdrift_rl

env = gym.make('VDrift-v0')
```

## Constructor

```python
VDriftEnv(
    render_mode=None,
    fromStartLine=5,
    vdrift_path=None,
    vdrift_cwd=None,
    redis_host='localhost',
    redis_port=6379,
    use_redis=True,
    resolution=(128, 128)
)
```

### Parameters

- **render_mode** (`str`, optional): Rendering mode
  - `None`: No rendering (default)
  - `'text'`: Text-based information
  - `'rgb_array'`: Return RGB array from `render()`

- **fromStartLine** (`int`, default=5): Starting line offset parameter

- **vdrift_path** (`str`, optional): Path to VDrift executable
  - If `None`, uses `$VDRIFT_BIN` environment variable
  - Falls back to `../../vdrift/build/vdrift` (relative to package)

- **vdrift_cwd** (`str`, optional): Working directory for VDrift
  - If `None`, auto-detected from vdrift_path parent directory

- **redis_host** (`str`, default='localhost'): Redis server hostname

- **redis_port** (`int`, default=6379): Redis server port

- **use_redis** (`bool`, default=True): Whether to use Redis for distributed instances
  - Set to `False` if Redis is not available or not needed

- **resolution** (`tuple`, default=(128, 128)): Observation image resolution (width, height)
  - Common values: `(128, 128)`, `(256, 256)`, `(64, 64)`

### Example

```python
env = gym.make('VDrift-v0',
               vdrift_path='/custom/path/to/vdrift',
               use_redis=False,
               resolution=(256, 256))
```

## Observation Space

**Type**: `gym.spaces.Box`

**Shape**: `(H, W, 3)` where H and W are from resolution parameter (default: 128×128)

**Data Type**: `np.uint8`

**Range**: `[0, 255]` for each RGB channel

**Description**: First-person view from the driver's perspective as an RGB image.

### Observation Details

- **Format**: RGB image (not BGR)
- **Source**: Rendered by VDrift's graphics engine
- **Frequency**: Updated every timestep (typically 50 Hz with dt=0.02)
- **Content**: Shows track, other cars, environment

## Action Space

**Type**: `gym.spaces.Box`

**Shape**: `(5,)`

**Data Type**: `np.float32`

### Action Vector Components

| Index | Control | Range | Description |
|-------|---------|-------|-------------|
| 0 | Throttle/Brake | [-0.9, 1.0] | Combined throttle/brake. Positive=throttle, negative=brake |
| 1 | Clutch | [0, 1] | Clutch pedal. 0=released, 1=fully pressed |
| 2 | Reserved | [0, 1] | Currently unused |
| 3 | Reserved | [0, 1] | Currently unused (was handbrake) |
| 4 | Steering | [-1, 1] | Steering angle. -1=full left, 0=center, 1=full right |

### Action Notes

- **Throttle/Brake Combined**: Action[0] uses positive values for throttle, negative for braking
  - The environment internally splits this: `throttle = max(action[0], 0)`, `brake = abs(min(action[0], 0))`
- **Handbrake**: Disabled in AI mode for simplified action space
- **Automatic Transmission**: Gear changes are handled automatically
- **Continuous Control**: All actions are continuous values (not discrete)

### Example Actions

```python
# Full throttle, center steering
action = [1.0, 0.0, 0.0, 0.0, 0.0]

# Full brake, turn right
action = [-0.9, 0.0, 0.0, 0.0, 0.5]

# Moderate throttle, slight left turn
action = [0.5, 0.0, 0.0, 0.0, -0.2]
```

## Reward Function

The reward function is designed to encourage fast, smooth racing while staying on track.

### Reward Components

```python
reward = progress_reward + velocity_reward + lap_reward - penalties
```

#### 1. Progress Reward
```python
progress_reward = moving_forward + (max_distance_multiplier * distance_gain)
```
- `moving_forward`: Distance moved along track since last step
- `distance_gain`: New maximum distance achieved this episode
- `max_distance_multiplier`: 10.0 if on track, 1.0 if off track

#### 2. Velocity Reward
```python
vel_reward = log(2.7183 + velocity_magnitude) * 0.04  # for higher speeds
# or
vel_reward = (velocity / minimal_speed) * timestep    # for low speeds
```
- Encourages maintaining speed
- Reduced when off-track (multiplier 0.5)

#### 3. Lap Reward
```python
lap_reward = 1000.0  # When completing a lap
```

#### 4. Penalties
- **Collision**: -5.0 per collision
- **Off-track**: -0.02 per timestep off track
- **Track Center Deviation**: Optional penalty based on distance from track center

### Reward Range

**Declared**: `(-40, 500)`

**Typical Values**:
- Normal progress: 0-10 per step
- Collision: ~-5
- Off-track: Small negative
- Lap completion: +1000

## Info Dictionary

The `info` dictionary returned by `step()` contains detailed telemetry:

```python
info = {
    'too_big_jump': bool,          # Teleportation detected
    'distance': float,             # Current distance along track (m)
    'max_dist': float,             # Maximum distance this episode (m)
    'out_of_center': float,        # Distance from track centerline (m)
    'velocity': float,             # Speed magnitude (m/s)
    'vel_sum_timestep_discounted': float,
    'moving_forward': float,       # Distance gain this step
    'max_distance_gain': float,    # Best distance gain
    'position': [x, y, z],         # Car 3D position
    'track_pos': [x, y, z],        # Nearest track centerline point
    'too_slow': bool,              # Speed below threshold
    'colided': bool,               # Collision this step
    'out_of_track': bool,          # Off-track indicator
    'ended': bool,                 # Episode termination flag
}
```

## Episode Termination

Episodes terminate (`done=True`) when:

1. **Too long out of track or slow**: Internal counter exceeds 2000
   - Incremented by: +5 if too slow, +80 if collision, +0 if off-track
   - Decremented by: -0.5 when driving normally

2. **Teleportation/jump detected**: Distance change > 400m in one step

3. **Lap completion**: After completing 2nd lap (configurable)

## Methods

### reset()

```python
observation = env.reset(seed=None, options=None)
```

**Returns**: Initial observation (RGB image)

**Side Effects**:
- Resets car to starting position
- Clears episode statistics
- Reconnects to VDrift if connection lost

### step()

```python
observation, reward, done, truncated, info = env.step(action)
```

**Parameters**:
- `action`: numpy array of shape (5,) with action values

**Returns**:
- `observation`: RGB image (H, W, 3)
- `reward`: float reward value
- `done`: bool indicating episode end
- `truncated`: bool (same as done in this environment)
- `info`: dict with telemetry data

### render()

```python
image = env.render(mode='rgb_array')
```

**Returns**: Current observation image if mode='rgb_array'

### close()

```python
env.close()
```

Closes socket connection to VDrift and cleans up resources.

## Advanced Configuration

### Using Environment Variables

```bash
export VDRIFT_BIN=/path/to/vdrift/build/vdrift
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Multiple Parallel Environments

```python
from stable_baselines3.common.env_util import make_vec_env

# Creates 4 parallel environments
env = make_vec_env(
    lambda: gym.make('VDrift-v0', use_redis=True),
    n_envs=4
)
```

**Note**: Redis helps manage VDrift instances when using multiple parallel environments.

### Custom Resolution for Training

```python
# Lower resolution for faster training
env = gym.make('VDrift-v0', resolution=(64, 64))

# Higher resolution for evaluation
env = gym.make('VDrift-v0', resolution=(256, 256))
```

## Wrappers

Common wrappers to use with VDriftEnv:

### Frame Stacking

```python
from stable_baselines3.common.vec_env import VecFrameStack

env = make_vec_env(lambda: gym.make('VDrift-v0'), n_envs=1)
env = VecFrameStack(env, n_stack=8)  # Stack 8 frames
```

### Observation Normalization

```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(env, norm_obs=False, norm_reward=True)
```

### Image Transpose (for PyTorch CNN)

```python
from stable_baselines3.common.vec_env import VecTransposeImage

env = VecTransposeImage(env)  # Converts (H,W,C) to (C,H,W)
```

## Performance Considerations

### Timestep

Default timestep is 0.02 seconds (50 Hz). This balances:
- Physics simulation accuracy
- Agent reaction time
- Training speed

### Resolution

Observation resolution affects:
- **Training speed**: Lower resolution = faster
- **Visual detail**: Higher resolution = more information
- **Memory usage**: Scales with width × height

Recommended: Start with 128×128, increase if needed.

### Network Latency

TCP communication adds ~1-5ms latency. For lowest latency:
- Run VDrift and Python on same machine
- Use localhost connection
- Disable unnecessary visual effects in VDrift

## Debugging

### Enable Verbose Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)

env = gym.make('VDrift-v0')
```

### Inspect Observations

```python
import matplotlib.pyplot as plt

obs = env.reset()
plt.imshow(obs)
plt.show()
```

### Monitor Info Dict

```python
obs, reward, done, truncated, info = env.step(action)
print(f"Position: {info['position']}")
print(f"Speed: {info['velocity']:.2f} m/s")
print(f"Distance: {info['distance']:.2f} m")
```

## Examples

See the [examples/](../examples/) directory for complete training scripts and usage examples.
