import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

# Footsteps Planning Environments
register(
    id="footsteps-planning-right-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightEnv",
    max_episode_steps=50,
)

register(
    id="footsteps-planning-left-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftEnv",
    max_episode_steps=50,
)

register(
    id="footsteps-planning-any-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyEnv",
    max_episode_steps=50,
)

register(
    id="footsteps-planning-right-withball-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightWithBallEnv",
    max_episode_steps=50,
)

register(
    id="footsteps-planning-left-withball-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftWithBallEnv",
    max_episode_steps=50,
)

register(
    id="footsteps-planning-any-withball-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyWithBallEnv",
    max_episode_steps=50,
)

register(
    id="footsteps-planning-right-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightMultiGoalEnv",
    max_episode_steps=70,
)

register(
    id="footsteps-planning-left-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftMultiGoalEnv",
    max_episode_steps=70,
)

register(
    id="footsteps-planning-any-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyMultiGoalEnv",
    max_episode_steps=70,
)

register(
    id="footsteps-planning-right-withball-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightWithBallMultiGoalEnv",
    max_episode_steps=90,
)

register(
    id="footsteps-planning-left-withball-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftWithBallMultiGoalEnv",
    max_episode_steps=90,
)

register(
    id="footsteps-planning-any-withball-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyWithBallMultiGoalEnv",
    max_episode_steps=90,
)

register(
    id="footsteps-planning-right-obstacle-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightObstacleMultiGoalEnv",
    max_episode_steps=90,
)

register(
    id="footsteps-planning-left-obstacle-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningLeftObstacleMultiGoalEnv",
    max_episode_steps=90,
)

register(
    id="footsteps-planning-any-obstacle-multigoal-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyObstacleMultiGoalEnv",
    max_episode_steps=90,
)