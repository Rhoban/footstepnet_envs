[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# FootStepNet Envs : Footsteps Planning RL Environments for Fast On-line Bipedal Footstep Planning and Forecasting

<img src="https://github.com/user-attachments/assets/056cf809-544b-4a97-b35f-bb06a846bae7" align="right" width="50%"/>

These environments are dedicated to train efficient agents that can plan and forecast bipedal robot footsteps in order to go to a target location possibly avoiding obstacles.
They are designed to be used with Reinforcement Learning (RL) algorithms (as implemented in [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)).

An example of a trained *FootstepNet* use:

- **Step 1**: A bipedal robot must score a goal while minimizing its number of steps. To do this, we arbitrarily choose $n_{alt}$ placement possibilities (here $n_{alt}=3$) which all allow scoring.
- **Step 2**: Forecasting allows choosing from the $n_{alt}$ possibilities, the one that requires the fewest steps.
- **Step 3**: The planner compute all the steps in order to go to the position chosen by the forecast.
- **Step 4**: The step sequence is executed on the real robot.

Consult the associated article for more information : [FootstepNet: an Efficient Actor-Critic Method for Fast On-line Bipedal Footstep Planning and Forecasting](https://arxiv.org/pdf/2403.12589)

## Installation

### Footsteps Planning Environments

From source:

```
pip install -e .
```

To train and enjoy using Stable Baselines3 (SB3), install RL Baselines3 Zoo:

```
pip install rl_zoo3
```

To train and enjoy using Stable Baselines Jax (SBX):

```
pip install sbx-rl
```

## Train the Agent

### Using RL Baselines3 Zoo and Stable Baselines3 (SB3)

The easiest way to train the agent is to use [RL Baselines3 Zoo](https://rl-baselines3-zoo.readthedocs.io/)

The hyperparameters for the environment are defined in `hyperparameters/[algo-name].yml`.
For now, the best DRL algorithm for this environment is TD3.

You can train an agent using:

```bash
python -m rl_zoo3.train \
    --algo td3 \
    --env footsteps-planning-right-v0 \
    --gym-packages gym_footsteps_planning \
    --conf hyperparams/td3.yml
```

Where:

- `--algo td3` is the RL algorithm to use (TD3 in this case).
- `--env footsteps-planning-right-v0` is the environment to train on (see **Environments** section).
- `--gym-packages gym_footsteps_planning` is used to register the environment.
- `--conf ./hyperparams/td3.yml` is the hyperparameters file to use.

The trained agent will be stored in the `.\logs\[algo-name]\[env-name]_[exp-id]` folder from the current working directory.

### Using Stable Baselines Jax (SBX)

```bash
python train_sbx.py \
    --algo crossq \
    --env footsteps-planning-right-v0 \
    --conf hyperparams/crossq.yml
```

## Enjoy a Trained Agent

### Using RL Baselines3 Zoo and Stable Baselines3 (SB3)

If a trained agent exists, you can see it in action using:

```bash
python -m rl_zoo3.enjoy \
    --algo td3 \
    --env footsteps-planning-right-v0 \
    --gym-packages gym_footsteps_planning \
    --folder logs/ \
    --load-best \
    --exp-id 0
```

Where:

- `--algo td3` is the RL algorithm to use (TD3 in this case).
- `--env footsteps-planning-right-v0` is the environment to enjoy on (see **Environments** section).
- `--gym-packages gym_footsteps_planning` is used to register the environment.
- `--folder logs/` is the folder where the trained agent is stored.
- `--load-best` is used to load the best agent.
- `--exp-id 0` is the experiment ID to use (`0` meaning the latest).

### Using Stable Baselines Jax (SBX)

```bash
python enjoy_sbx.py \
    --algo crossq \
    --env footsteps-planning-right-v0 \
    --gym-packages gym_footsteps_planning \
    --folder logs/ \
    --load-best \
    --exp-id 0
```

## Environments

These environments were first design to play soccer with humanoids robots (see [RoboCup Humanoid League](https://www.robocup.org/leagues/3)). Indeed, they are made designed to place the robot in front of a ball as long as not walking on it (to shoot for example). Or even avoid an obstacle (an opponent for example) while going to a specific location.

Each environment is available in 3 different versions :

- *Right* : The target during training is always the right foot.
- *Left* : The target during training is always the left foot.
- *Any* : The target during training is either the left or the right foot (with 0.5 probability for each). It means that the trained agent can then have either foot as target.

### Action Space, Observation Space and Reward

The action and observation spaces, as well as the reward are common to all environments.

#### Observation Space

Num | Observation | Min | Max
---|---|---|---
0 | x Target support foot position [m] | $-\sqrt{4^2+4^2}$ | $\sqrt{4^2+4^2}$
1 | y Target support foot position [m] | $-\sqrt{4^2+4^2}$ | $\sqrt{4^2+4^2}$
2 | cos(theta) target support foot orientation | -1 | 1
3 | sin(theta) target support foot orientation | -1 | 1
4 | Is the current foot the target foot ? | 0 | 1

If obstacle is enabled (see below), the following observations are added:

Num | Extra observations with obstacle  | Min | Max
---|---|---|---
5 | x obstacle position in the frame of the foot [m] | $-\sqrt{4^2+4^2}$ | $\sqrt{4^2+4^2}$
6 | y obstacle position in the frame of the foot [m] | $-\sqrt{4^2+4^2}$ | $\sqrt{4^2+4^2}$
7 | Obstacle radius [m] | 0 | 0.25

**Note: The observation space positions are all defined in the frame of the current support foot. If the support foot is the left foot, transformations are used to ensure sagital symmetry.**

#### Action Space

Num | Action | Min | Max
---|---|---|---
0 | Non-support foot movement along the x axis [m] | -0.08* | 0.08
1 | Non-support foot movement along the y axis [m] | -0.04 | 0.04
2 | Non-support foot rotation [deg] | -20 | 20

*: Maximum forward step is used here to ensure a zero-centered action space. However, the backward step is clipped to 0.04 to ensure the robot stability.

#### Reward

The reward is defined as follows:

$$ R = - \delta_\text{distance error} \times 0.1 - \delta_\text{angle error} \times 0.05 - \delta_\text{collision}$$

Where:

- $\delta_\text{distance error}$ is the distance error between the target foot position and the current foot position.
- $\delta_\text{angle error}$ is the angle error between the target foot orientation and the current foot orientation.
- $\delta_\text{collision}$ is equal to **10** if the foot is colliding with the obstacle, else it is equal to **1** (penalty for each step taken) .

### Options

Below are the customizable options for the `FootstepsPlanningEnv` environment:

| Option | Description | Default Value |
|--------|-------------|---------------|
| `max_dx_forward` | Maximum forward step size [m] | `0.08` |
| `max_dx_backward` | Maximum backward step size [m] | `0.03` |
| `max_dy` | Maximum lateral step size [m] | `0.04` |
| `max_dtheta` | Maximum rotation step size [rad] | `np.deg2rad(20)` |
| `tolerance_distance` | Distance tolerance for reaching the goal [m] | `0.05` |
| `tolerance_angle` | Angle tolerance for reaching the goal [rad] | `np.deg2rad(5)` |
| `has_obstacle` | Whether the environment includes an obstacle | `False` |
| `obstacle_max_radius` | Maximum radius of the obstacle [m] | `0.25` |
| `obstacle_radius` | Fixed radius of the obstacle, or `None` for random | `None` |
| `obstacle_position` | Position of the obstacle [m, m] | `np.array([0, 0])` |
| `foot` | Target foot for the agent (`"any"`, `"left"`, or `"right"`) | `"any"` |
| `foot_length` | Length of the foot [m] | `0.14` |
| `foot_width` | Width of the foot [m] | `0.08` |
| `feet_spacing` | Spacing between feet [m] | `0.15` |
| `shaped` | Whether to include a reward shaping term | `True` |
| `multi_goal` | If `True`, the goal is sampled in a 4x4 m area, otherwise fixed at `[0, 0]` | `False` |

### Placer without obstacle/ball

<img src="https://github.com/user-attachments/assets/56a64d7d-7ff5-4c93-9832-0f8c3c0b9b03" align="right" width="25%"/>

#### Environment names

- Right foot as target: `footsteps-planning-right-v0`
- Left foot as target: `footsteps-planning-left-v0`
- Alternating feet as target: `footsteps-planning-any-v0`

#### Description

This environment allows to train an agent that place the desired foot of the robot to a specific location.

#### Starting State

The starting foot and the starting foot pose are randomly generated at each episode within a defined range (cf. Observation state).

#### Goal State

The target foot is fixed (*right* or *left*) or randomly generated (*any*) at each episode. The target foot pose is fixed.

### Placer with a ball

<img src="https://github.com/user-attachments/assets/96818653-0847-444f-935f-5e224c357a11" align="right" width="25%"/>

#### Environment names

- Right foot as target: `footsteps-planning-right-withball-v0`
- Left foot as target: `footsteps-planning-left-withball-v0`
- Alternating feet as target: `footsteps-planning-any-withball-v0`

#### Description

This environment allows to train an agent that place the desired foot of the robot to a specific location while avoiding an obstacle of a fixed size (for example a ball).

#### Starting State

The starting foot and the starting foot pose are randomly generated at each episode within a defined range (cf. Observation state). A fixed-size obstacle is present ([0.3,0] in the world frame).

#### Goal State

The target foot is fixed (*right* or *left*) or randomly generated (*any*) at each episode. The target foot pose is fixed and in front of the obstacle ([0,0] in the world frame).

### Multi-goal placer without obstacle/ball

<img src="https://github.com/user-attachments/assets/2980adc3-eecc-4f2c-a60b-cca85eda9aaa" align="right" width="25%"/>

#### Environment names

- Right foot as target: `footsteps-planning-right-multigoal-v0`
- Left foot as target: `footsteps-planning-left-multigoal-v0`
- Alternating feet as target: `footsteps-planning-any-multigoal-v0`

#### Description

This environment allows to train an agent that place the desired foot of the robot to a different location at each episode.

#### Starting State

The starting foot and the starting foot pose are randomly generated at each episode within a defined range (cf. Observation state).

#### Goal State

The target foot is fixed (*right* or *left*) or randomly generated (*any*) at each episode. The target foot pose is randomly generated within a defined range (cf. Observation state).

### Multi-goal placer with a ball

<img src="https://github.com/user-attachments/assets/f6256f1e-a478-4d5d-a076-b98294327eb2" align="right" width="25%"/>

#### Environment names

- Right foot as target: `footsteps-planning-right-withball-multigoal-v0`
- Left foot as target: `footsteps-planning-left-withball-multigoal-v0`
- Alternating feet as target: `footsteps-planning-any-withball-multigoal-v0`

#### Description

This environment allows to train an agent that place the desired foot of the robot to a different location at each episode while avoiding an obstacle of a fixed size (for example a ball).

#### Starting State

The starting foot and the starting foot pose are randomly generated at each episode within a defined range (cf. Observation state). A fixed-size obstacle is present ([0.3,0] in the world frame).

#### Goal State

The target foot is fixed (*right* or *left*) or randomly generated (*any*) at each episode. The target foot pose is randomly generated within a defined range (cf. Observation state).

### Multi-goal placer with size-variable obstacle

<img src="https://github.com/user-attachments/assets/4ac73b7a-3325-441f-9fad-47998c0969ed" align="right" width="25%"/>

#### Environment names

- Right foot as target: `footsteps-planning-right-obstacle-multigoal-v0`
- Left foot as target: `footsteps-planning-left-obstacle-multigoal-v0`
- Alternating feet as target: `footsteps-planning-any-obstacle-multigoal-v0`

#### Description

This environment allows to train an agent that place the desired foot of the robot to a different location at each episode while avoiding an obstacle of a variable size.

#### Starting State

The starting foot and the starting foot pose are randomly generated at each episode within a defined range (cf. Observation state). An obstacle is present in the environment ([0.3,0] in the world frame) and its size is randomly generated at each episode.

#### Goal State

The target foot is fixed (*right* or *left*) or randomly generated (*any*) at each episode. The target foot pose is randomly generated within a defined range (cf. Observation state).

## Citing the Project

To cite this repository in publications:

```bibtex
@article{footstepnet,
  title={FootstepNet: an Efficient Actor-Critic Method for Fast On-line Bipedal Footstep Planning and Forecasting},
  author={Gaspard, Cl{\'e}ment and Passault, Gr{\'e}goire and Daniel, M{\'e}lodie and Ly, Olivier},
  journal={arXiv preprint arXiv:2403.12589},
  year={2024}
}
```

**Note** : The environments were tested with the following packages version :

```
gymnasium==0.29.1 numpy==1.26.4 stable_baselines3==2.3.2 sb3_contrib==2.3.0 pygame==2.6.0
```
