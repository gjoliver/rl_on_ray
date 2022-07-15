import ray

from model import Model
from simulator import Simulator


NUM_SIMULATORS = 5
NUM_ITERATIONS = 100
HEALTH_CHECK_TIMEOUT = 0.5  # 0.5 sec.


def train_step(simulators, trainable):
    """Single train step.
    """
    # Synchronous rollout.
    # Collect 1 episode from each simulator and feed those to trainable.
    # TODO: support Asynchronous rollout.
    episode_refs = [s.sample.remote() for s in simulators]
    episodes = ray.get(episode_refs)

    # Train local model with collected episodes.
    return trainable.train(episodes)


def sync_weights(simulators, trainable):
    weights = trainable.get_weights()
    # Manually create obj_ref to avoid serializing weights multiple times.
    weights_ref = ray.put(weights)
    ray.wait([s.set_weights.remote(weights_ref) for s in simulators])


def try_recover_simulators(simulators, trainable):
    # Figure out which simulator is faulty.
    faulty = []
    for i, s in enumerate(simulators):
        try:
            healthy = ray.get(s.is_healthy.remote(), timeout=HEALTH_CHECK_TIMEOUT)
            if not healthy: faulty.append(i)
        except Exception:
            faulty.append(i)

    # Try to recreate the simulators.
    for i in faulty:
        print(f"Recovering simulator {i} ...")

        bad_simulator = simulators[i]
        simulators[i] = Simulator.remote(i)
        del bad_simulator


@ray.remote
def run():
    ray.init()

    # Distributed rollout.
    simulators = [Simulator.remote(i) for i in range(NUM_SIMULATORS)]
    # Make sure all simulators are up and running.
    assert all([s.is_healthy.remote() for s in simulators])

    # Simulators have their own copies of model for rollout / inferencing.
    # This is the Local model that we actually train on.
    trainable = Model()

    i = 0
    episodes = 0
    timesteps = 0
    while i < NUM_ITERATIONS:
        if i % 10 == 0:
            print(f"Iteration {i}, {episodes} episodes, {timesteps} timesteps.")

        try:
            # Train.
            result = train_step(simulators, trainable)
            episodes += result["num_episodes"]
            timesteps += result["num_timesteps"]

            # Assuming on-policy training, sync weights after each step.
            sync_weights(simulators, trainable)
        except Exception as e:
            # Something went wrong.
            print(e)

            # Check health and try to bring failed simulators back.
            try_recover_simulators(simulators, trainable)

        i += 1

    ray.shutdown()


if __name__ == "__main__":
    run.remote()
