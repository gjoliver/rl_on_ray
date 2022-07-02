import gym
import numpy as np
import pandas as pd
import random
import ray

from model import Model, OBS_DIM


NUM_CPUS = 1
NUM_GPUS = 0


# Simulator that is responsible for distributed rollout.
@ray.remote(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
class Simulator:
    def __init__(self, i):
        self._i = i
        self._env = gym.make("CartPole-v0")
        self._model = Model()
        self._is_healthy = True

    def sample(self):
        """Sample and return an episode.

        Note: may throw exceptions if anything goes wrong.
        """
        # Simulate the scenario where things go wrong sometimes.
        if random.random() < 0.01:
            self._is_healthy = False
            raise Exception(f"Simulator {self._i} failed.")

        obs = []
        actions = []
        rewards = []
        dones = []
        infos = []

        observation = self._env.reset()
        reward = 0.0
        done = False
        info = {}
        while not done:
            obs.append(observation)

            action = self._model.predict(
                # Add batch dimension.
                np.array(observation).reshape((1, OBS_DIM))
            )
            observation, reward, done, info = self._env.step(action)

            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return pd.DataFrame({
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
        })

    def set_weights(self, weights):
        self._model.set_weights(weights)

    def is_healthy(self):
        return self._is_healthy
