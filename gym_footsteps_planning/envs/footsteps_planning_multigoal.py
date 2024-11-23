from .footsteps_planning_env import FootstepsPlanningEnv


# Normal envs
class FootstepsPlanningRightMultiGoalEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["shaped"] = True
        options["foot"] = "right"
        options["has_obstacle"] = False
        options["multi_goal"] = True

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)


class FootstepsPlanningLeftMultiGoalEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["shaped"] = True
        options["foot"] = "left"
        options["has_obstacle"] = False
        options["multi_goal"] = True

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)


class FootstepsPlanningAnyMultiGoalEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["shaped"] = True
        options["has_obstacle"] = False
        options["multi_goal"] = True

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)
