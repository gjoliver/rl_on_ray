from tensorflow import keras
from tensorflow.keras import layers


OBS_DIM = 4
ACTION_DIM = 2
HIDDENS = [32, 32]


# Example model. This is managed by TFAgent in reality.
class Model:
    def __init__(self):
        # Create keras model.
        self._model = keras.Sequential(
            [
                keras.Input(shape=(OBS_DIM,), name="obs")
            ] + [
                layers.Dense(h, activation="relu") for h in HIDDENS
            ] + [
                layers.Dense(ACTION_DIM)
            ])

    def train(self, episodes):
        # Not gonna do actual training with this example.
        return {
            "num_episodes": len(episodes),
            "num_timesteps": sum([len(e) for e in episodes]),
        }

    def predict(self, obs):
        logits = self._model(obs)
        actions = keras.backend.argmax(
            keras.activations.softmax(logits)
        )
        return actions[0].numpy()

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)
