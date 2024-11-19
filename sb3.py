import time
import gymnasium
import argparse
import numpy as np
import gym_footsteps_planning

from stable_baselines3 import TD3, A2C, SAC, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

parser = argparse.ArgumentParser(description="SB3.")
parser.add_argument("-m", "--model", required=False, type=str, default="td3")
parser.add_argument("-e", "--env", required=False, type=str, default="footsteps-planning-right-v0")
parser.add_argument("-x", "--experiment", required=True, type=str)
parser.add_argument("-l", "--load", action="store_true")
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-d", "--device", required=False, type=str, default="cpu")
parser.add_argument("-s", "--steps", required=False, type=int, default=10000000)
args = parser.parse_args()

classes = {
    "td3": TD3,
}

if args.model in classes:
    model_class = classes[args.model]
else:
    print("Unknown model: " + args.model)
    exit()

fname = "networks/" + args.experiment + "_" + args.env + "_" + args.model
env = gymnasium.make(args.env)

goal_selection_strategy = "future"


class ScheduledNormalActionNoise(ActionNoise):
    """
    A gaussian noise scheduled to decrease through time
    """

    def __init__(self, n_actions: int, sigma_start=0.2, sigma_end=0.02, annealing: int = 1000000):
        self._step = 0
        self._annealing = annealing
        self._mu = np.zeros(n_actions)
        self._sigma_start = sigma_start * np.ones(n_actions)
        self._sigma_end = sigma_end * np.ones(n_actions)
        super(ScheduledNormalActionNoise, self).__init__()

    def __call__(self) -> np.ndarray:
        self._step += 1
        alpha = np.clip(self._step / self._annealing, 0, 1)
        sigma = self._sigma_start + alpha * (self._sigma_end - self._sigma_start)

        return np.random.normal(self._mu, sigma)


n_actions = env.action_space.shape[-1]
action_noise = ScheduledNormalActionNoise(n_actions)

parameters = {"env": env, "action_noise": action_noise, "device": args.device, "verbose": 1, "train_freq": (5, "step")}


class SavePeriodicallyCallback(BaseCallback):
    def __init__(self, check_freq: int, fname: str, verbose: int = 1):
        super(SavePeriodicallyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.fname = fname

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print("Saving to %s..." % self.fname)
            self.model.save(self.fname)

        return True


if args.load:
    print("Loading %s" % fname)
    model = model_class.load(fname, **parameters)
else:
    print("Creating a new model")
    model = model_class(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=goal_selection_strategy,
        ),
        verbose=1,
    )


if args.train:
    print("Starting training")
    callback = SavePeriodicallyCallback(10000, fname)
    model.learn(total_timesteps=args.steps, log_interval=10, callback=callback)

state, infos = env.reset()
score = 0
for i in range(1000):
    action, _state = model.predict(state, deterministic=True)

    state, reward, done, truncated, infos = env.step(action)
    score += reward

    env.render()
    if done or truncated:
        state, infos = env.reset()
        print("Score: %f" % score)
        score = 0
