from .footsteps_planning_env import FootstepsPlanningEnv


# Normal envs
class FootstepsPlanningRightEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["foot"] = "right"
        options["has_obstacle"] = False

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)


class FootstepsPlanningLeftEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["foot"] = "left"
        options["has_obstacle"] = False

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)


class FootstepsPlanningAnyEnv(FootstepsPlanningEnv):
    def __init__(self, train=False, visualize=False, render_mode="human", options=None):
        options = options or {}
        options["has_obstacle"] = False

        super().__init__(train=train, visualize=visualize, render_mode=render_mode, options=options)
