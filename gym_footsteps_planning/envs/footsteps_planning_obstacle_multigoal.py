import numpy as np
from .footsteps_planning_env import FootstepsPlanningEnv


# Normal envs
class FootstepsPlanningRightObstacleMultiGoalEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["shaped"] = True
        options["foot"] = "right"
        options["has_obstacle"] = True
        options["obstacle_position"] = np.array([0.3, 0], dtype=np.float32)
        options["multi_goal"] = True

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)


class FootstepsPlanningLeftObstacleMultiGoalEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["shaped"] = True
        options["foot"] = "left"
        options["has_obstacle"] = True
        options["obstacle_position"] = np.array([0.3, 0], dtype=np.float32)
        options["multi_goal"] = True

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)


class FootstepsPlanningAnyObstacleMultiGoalEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["shaped"] = True
        options["has_obstacle"] = True
        options["obstacle_position"] = np.array([0.3, 0], dtype=np.float32)
        options["multi_goal"] = True

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)
