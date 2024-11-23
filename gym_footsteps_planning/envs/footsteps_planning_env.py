import gymnasium
import math
import numpy as np
from typing import Optional
from gymnasium import spaces
from gym_footsteps_planning.footsteps_simulator.simulator import Simulator as FootstepsSimulator
from gym_footsteps_planning.footsteps_simulator import transform as tr


class FootstepsPlanningEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self, options: Optional[dict] = None, train: bool = False, visualize: bool = False, render_mode: str = "none"
    ):
        self.options = {
            # Maximum steps
            "max_dx_forward": 0.08,  # [m]
            "max_dx_backward": 0.03,  # [m]
            "max_dy": 0.04,  # [m]
            "max_dtheta": np.deg2rad(20),  # [rad]
            # Target tolerance
            "tolerance_distance": 0.05,  # [m]
            "tolerance_angle": np.deg2rad(5),  # [rad]
            # Do we include collisions with the ball?
            "has_obstacle": False,
            "obstacle_max_radius": 0.25,  # [m]
            "obstacle_radius": None,  # [m]
            "obstacle_position": np.array([0, 0], dtype=np.float32),  # [m,m]
            # Which foot is targeted (any, left or right)
            "foot": "any",
            # Foot geometry
            "foot_length": 0.14,  # [m]
            "foot_width": 0.08,  # [m]
            "feet_spacing": 0.15,  # [m]
            # Add reward shaping term
            "shaped": True,
            # If True, the goal will be sampled in a 4x4m area, else it will be fixed at (0,0)
            "multi_goal": False,
        }
        self.options.update(options or {})

        # Render mode
        self.visualize: bool = visualize
        self.render_mode: str = render_mode

        self.simulator: FootstepsSimulator = FootstepsSimulator()
        self.simulator.feet_spacing = self.options["feet_spacing"]
        self.simulator.foot_length = self.options["foot_length"]
        self.simulator.foot_width = self.options["foot_width"]

        # Maximum speed in each dimension
        self.min_step = np.array(
            [-self.options["max_dx_backward"], -self.options["max_dy"], -self.options["max_dtheta"]], dtype=np.float32
        )
        self.max_step = np.array(
            [self.options["max_dx_forward"], self.options["max_dy"], self.options["max_dtheta"]], dtype=np.float32
        )

        # Action space is target step size (dx, dy, dtheta)
        # To keep 0 as "not moving", we use maxStep instead of maxBackwardStep,
        # but the speed is clipped when stepping
        self.action_high = np.array(
            [self.options["max_dx_forward"], self.options["max_dy"], self.options["max_dtheta"]], dtype=np.float32
        )
        self.action_low = -self.action_high

        self.action_space = spaces.Box(self.action_low, self.action_high)

        # State is position and orientation, here limited in a √(2*4²)x√(2*4²)m area arround the support foot
        # and the current step size
        # - x target support foot position in the frame of the foot
        # - y target support foot position in the frame of the foot
        # - cos(theta) target support foot orientation in the frame of the foot
        # - sin(theta) target support foot orientation in the frame of the foot
        # - is the current foot the target foot ?
        # - x obstacle position in the frame of the foot
        # - y obstacle position in the frame of the foot
        # - the obstacle radius
        max_diag_env = np.sqrt(2 * (4**2))
        self.state_low_goal = np.array(
            [-max_diag_env, -max_diag_env, -1, -1, 0, -max_diag_env, -max_diag_env, 0], dtype=np.float32
        )
        self.state_high_goal = np.array(
            [max_diag_env, max_diag_env, 1, 1, 1, max_diag_env, max_diag_env, self.options["obstacle_max_radius"]],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(self.state_low_goal, self.state_high_goal)

        self.reset(seed=0)

    def get_observation(self) -> np.ndarray:
        """
        Builds an observation from the current internal state
        """
        T_support_world = tr.frame_inv(tr.frame(*self.simulator.support_pose()))

        T_support_target = T_support_world @ tr.frame(*self.target_foot_pose)
        support_target = np.array(
            [
                T_support_target[0, 2],  # x
                T_support_target[1, 2],  # y
                T_support_target[0, 0],  # cos(theta)
                T_support_target[1, 0],  # sin(theta)
            ],
            dtype=np.float32,
        )

        if self.options["has_obstacle"]:
            self.support_obstacle = tr.apply(T_support_world, self.options["obstacle_position"])

        # Define if the target foot is the right one
        is_target_foot = 1 if (self.simulator.support_foot == self.target_support_foot) else 0

        # Handling symmetry
        if self.simulator.support_foot == "left":
            # Invert the target foot position and orientation for the other foot
            support_target[1] = -support_target[1]
            support_target[3] = -support_target[3]

            # Invert the obstacle position for the other foot if there is one
            if self.options["has_obstacle"]:
                self.support_obstacle[1] = -self.support_obstacle[1]

        # state = [
        #     *support_target,
        #     is_target_foot,
        # ] + ([*self.support_obstacle, self.obstacle_radius] if self.options["has_obstacle"] else [0, 0, 0])

        state = np.array(
            [
                *support_target,
                is_target_foot,
            ]
            + ([*self.support_obstacle, self.obstacle_radius] if self.options["has_obstacle"] else [0, 0, 0]),
            dtype=np.float32,
        )

        state = np.array(state, dtype=np.float32)

        return state

    def ellipsoid_clip(self, step: np.ndarray) -> np.ndarray:
        """
        Applying a rescale of the order in an "ellipsoid" manner. This transforms the target step to
        a point in a space where it should lie on a sphere, ensure its norm is not high than 1 and takes
        it back to the original scale.
        """
        factor = np.array(
            [
                self.options["max_dx_forward"] if step[0] >= 0 else self.options["max_dx_backward"],
                self.options["max_dy"],
                self.options["max_dtheta"],
            ],
            dtype=np.float32,
        )
        clipped_step = step / factor

        # In this space, the step norm should be <= 1
        norm = np.linalg.norm(clipped_step)
        if norm > 1:
            clipped_step /= norm

        return clipped_step * factor

    def step(self, action):
        """
        One step of the environment. Takes one step, checks for collisions and reached conditions.
        """
        if self.simulator.support_foot == "left":
            action[1] = -action[1]
            action[2] = -action[2]

        # Making sure action is clipped by its bounds
        step = np.clip(action, self.action_low, self.action_high)

        # Clipping the step
        clipped_step = self.ellipsoid_clip(step)

        # Taking the step in the simulator
        self.simulator.step(*clipped_step)

        # Retrieve the observation
        state = self.get_observation()

        # Distance between the achieved and desired goal

        distance = np.linalg.norm(state[:2])
        target_theta = self.target_foot_pose[2]
        theta_error = np.arccos(np.clip(np.dot(state[2:4], np.array([1, 0])), -1.0, 1.0), dtype=np.float32)

        is_desired_foot = state[4] == 1

        # Do we collide the target area ? (checking all the feet corners)
        in_obstacle = False
        if self.options["has_obstacle"]:
            support_obstacle = state[-3:-1]
            obstacle_radius = state[-1]

        if self.options["has_obstacle"]:
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    # One of the corners of feet is walking in the forbidden area, punishing this with negative reward
                    P_corner_foot = np.array(
                        [sx * self.simulator.foot_length / 2, sy * self.simulator.foot_width / 2], dtype=np.float32
                    )

                    if np.linalg.norm(P_corner_foot - support_obstacle) < obstacle_radius:
                        in_obstacle = True

        if distance < self.options["tolerance_distance"] and theta_error < self.options["tolerance_angle"] and is_desired_foot:
            # We reached the goal
            reward = 0
            terminated = True
        else:
            # We did not reach the goal, -10 if we are walking in obstacle, -1 else
            reward = -10 if in_obstacle else -1
            terminated = False

            # Reward shaping if needed
            if self.options["shaped"]:
                reward -= (distance + theta_error / 2.0) * 1e-1

        if self.visualize:
            self.do_render()

        return state, reward, terminated, False, {}

    def reset(self, seed: int | None = None, options: Optional[dict] = None):
        """
        Resets the environment to a given foot pose and support foot
        """
        # Seeding the environment
        super().reset(seed=seed)
        options = options or {}

        # Getting the options
        start_foot_pose = options.get("start_foot_pose", None)
        start_support_foot = options.get("start_support_foot", None)
        self.target_foot_pose = options.get("target_foot_pose", None)
        self.target_support_foot = options.get("target_support_foot", None)
        self.obstacle_radius = options.get("obstacle_radius", None)

        # Choosing obstacle radius and position
        if self.options["has_obstacle"]:
            if self.options["obstacle_radius"] is not None and self.obstacle_radius is None:
                self.obstacle_radius = self.options["obstacle_radius"]
            elif self.obstacle_radius is None:
                self.obstacle_radius = self.np_random.uniform(0, self.options["obstacle_max_radius"])

            self.simulator.clear_obstacles()
            self.simulator.add_obstacle(self.options["obstacle_position"], self.obstacle_radius)

        # Choosing starting foot
        if start_support_foot is None:
            start_support_foot = "left" if (self.np_random.uniform(0, 1) > 0.5) else "right"

        # Choosing target foot
        if self.target_support_foot is None:
            if self.options["foot"] != "any":
                self.target_support_foot = self.options["foot"]
            else:
                self.target_support_foot = "left" if (self.np_random.uniform(0, 1) > 0.5) else "right"

        # Sampling starting position and orientation
        if start_foot_pose is None:
            start_foot_pose = self.np_random.uniform([-2, -2, -math.pi], [2, 2, math.pi])

        # Initializing the simulator
        self.simulator.init(*start_foot_pose, start_support_foot)

        if self.target_foot_pose is None:
            if self.options["multi_goal"]:
                self.target_foot_pose = self.np_random.uniform([-2, -2, -math.pi], [2, 2, math.pi])
            else:
                self.target_foot_pose = np.array([0, 0, 0], dtype=np.float32)

        self.simulator.set_desired_goal(*self.target_foot_pose, self.target_support_foot)

        return self.get_observation(), {}

    def render(self):
        """
        Renders the footsteps
        """
        self.visualize = True

    def do_render(self):
        self.simulator.render()
