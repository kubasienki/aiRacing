# VDrift Setup and Modifications

This document describes how to build the modified VDrift simulator and details the changes made to support reinforcement learning.

## Building VDrift

### Prerequisites

VDrift requires several development libraries. Install them first:

#### Ubuntu/Debian
```bash
sudo apt install -y \
    build-essential scons \
    libsdl2-dev libsdl2-image-dev libsdl2-net-dev \
    libvorbis-dev libgl1-mesa-dev libglu1-mesa-dev \
    libarchive-dev libcurl4-openssl-dev \
    libbullet-dev libglew-dev
```

#### Arch Linux
```bash
sudo pacman -S base-devel scons \
    sdl2 sdl2_image sdl2_net \
    libvorbis mesa glu libarchive curl \
    bullet glew
```

### Build Steps

1. Navigate to the vdrift directory:
```bash
cd vdrift
```

2. Build using SCons:
```bash
scons
```

For a release build with optimizations:
```bash
scons release=1
```

For parallel build (faster):
```bash
scons -j$(nproc)
```

3. The executable will be created at:
```
vdrift/build/vdrift
```

### Build Options

Common SCons build options:

```bash
# Debug build with symbols
scons debug=1

# Release build (optimized)
scons release=1

# Minimal build
scons minimal=1

# Force rebuild
scons -c  # clean
scons     # build
```

## Modifications for RL

The VDrift codebase has been modified to support reinforcement learning through a TCP-based communication protocol. Here are the key changes:

### 1. AI TCP Server (`src/aitcpserver.h`, `src/aitcpserver.cpp`)

**New Files**: TCP server infrastructure for agent communication.

**Purpose**: Allows external RL agents to connect and control the car.

**Protocol**:
- Listens on a configurable port (default: 8081)
- Receives action commands from agent
- Sends back state observations and telemetry

### 2. Telemetry Data Structure (`src/aitcpserver.h`)

```cpp
struct aiInfoData {
    Vec3 position;           // Car position (x, y, z)
    Vec3 velocity;           // Car velocity vector
    float distance;          // Distance traveled along track
    float out_of_track;      // Boolean: off-track indicator
    Vec3 waypoint[5];        // Next 5 track waypoints
    float colided;           // Collision indicator
    float laps;              // Lap counter (ADDED FOR RL)
};
```

**Key Addition**: `laps` field enables multi-lap training episodes.

### 3. Action Data Structure (`src/aitcpserver.h`)

```cpp
struct aiStepData {
    float dt;                // Timestep
    float throttle;          // Throttle input [0, 1]
    float clutch;            // Clutch input [0, 1]
    float brake;             // Brake input [0, 1]
    float handbrake;         // Handbrake input [0, 1]
    float steering;          // Steering input [-1, 1]
    bool reset;              // Reset signal
    bool screenshot;         // Request screenshot
    bool fromStartLine;      // Start from start line
};
```

### 4. Game Loop Integration (`src/game.cpp`)

**Changes**:

a) **Lap Counter Integration** (line ~868):
```cpp
float laps = timer.GetCurrentLap(0) * 1.0;
data.laps = laps;  // Add lap count to telemetry
```

b) **Steering Input Mapping** (line ~1424):
```cpp
// Fixed mapping for left/right steering
carinputs[CarInput::STEER_LEFT] = -std::min(lastAiStep.steering, 0.0f);
carinputs[CarInput::STEER_RIGHT] = std::max(lastAiStep.steering, 0.0f);
```

c) **Handbrake Disabled** (line ~1427):
```cpp
carinputs[CarInput::HANDBRAKE] = 0.0;  // Simplified action space
```

d) **AI Mode Initialization** (line ~227):
```cpp
NewGame(false, false, 4);  // Start with 4 opponents for realistic racing
```

### 5. Resolution Configuration (`src/settings.cpp`)

**Changes** (line ~131):
```cpp
resolution[0] = 800;  // Changed from 128
resolution[1] = 800;  // Changed from 128
```

**Purpose**: Higher default resolution for better visual quality. The RL environment can override this via command-line arguments.

## TCP Communication Protocol

### Connection Flow

1. **Server Start**: VDrift starts TCP server on specified port
2. **Client Connect**: Python environment connects via socket
3. **Episode Loop**:
   - Client sends action packet
   - Server processes game step (dt seconds)
   - Server sends state packet
   - Server optionally sends screenshot image
4. **Episode Reset**: Client sends reset command

### Data Packets

#### Action Packet (Python → VDrift)
```
Format: "ffffff???" (27 bytes)
- dt (float): timestep duration
- throttle (float): 0.0-1.0
- clutch (float): 0.0-1.0
- brake (float): 0.0-1.0
- handbrake (float): 0.0-1.0
- steering (float): -1.0 to 1.0
- reset (bool): trigger episode reset
- screenshot (bool): request RGB image
- fromStartLine (bool): reset to start line
```

#### State Packet (VDrift → Python)
```
Format: "fffffffffffffffffffffffffffffff" (124 bytes)
31 floats including:
- position (x, y, z)
- velocity (vx, vy, vz)
- distance along track
- track boundaries
- waypoints (5 sets of x, y, z)
- collision indicator
- lap count
```

#### Screenshot Data (VDrift → Python)
```
1. Size message: "SIZE:XXXXX"
2. Raw RGB bytes: width * height * 3 bytes
3. Format: RGB888, row-major order
```

## Running VDrift in AI Mode

### Command-Line Arguments

```bash
./vdrift -ai <port>                    # Enable AI mode on port
./vdrift -ai <port> -nosound           # Disable sound
./vdrift -ai <port> -resolution 128x128  # Set resolution
./vdrift -ai <port> -multithreaded     # Enable multithreading
```

### Example

```bash
cd vdrift
./build/vdrift -ai 8081 -nosound -multithreaded -resolution 128x128
```

## Testing the Connection

Use the provided test client:

```bash
cd tools
python client.py
```

This will connect to VDrift and send random actions, useful for verifying the setup.

## Troubleshooting

### Build Errors

**Missing dependencies**:
```
Error: sdl2 not found
```
→ Install libsdl2-dev

**Bullet physics errors**:
```bash
sudo apt install libbullet-dev
```

### Runtime Errors

**Port already in use**:
```
socket bind failed
```
→ Change port or kill existing process:
```bash
lsof -ti:8081 | xargs kill
```

**Cannot connect to display**:
```
SDL error: No available video device
```
→ Use Xvfb for headless:
```bash
xvfb-run ./vdrift -ai 8081
```

## Development

### Making Further Modifications

1. Edit source files in `vdrift/src/`
2. Rebuild: `scons`
3. Test with `tools/client.py`
4. Commit changes to your fork

### Key Files for RL Development

- `src/aitcpserver.{h,cpp}` - TCP communication
- `src/game.cpp` - Main game loop and telemetry
- `src/car.cpp` - Vehicle dynamics
- `src/track.cpp` - Track information

## References

- [VDrift Official Documentation](http://wiki.vdrift.net/)
- [VDrift GitHub](https://github.com/VDrift/vdrift)
- [VDrift Build Guide](http://wiki.vdrift.net/Compiling)
