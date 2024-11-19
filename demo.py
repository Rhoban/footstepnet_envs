import time
import gym_footsteps_planning
import gymnasium

env = gymnasium.make("footsteps-planning-right-v0")
env.reset()
step = 0

while True:
    state, reward, done, truncated, infos = env.step([0.1, 0.0, 0.2])
    step += 1

    print(f"STEP [{step}]")
    print(f"State: {state}")
    print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")

    env.render()

    if done or truncated:
        step = 0
        env.reset()

    time.sleep(0.05)
