"""
    Hello ** Keras ** World!

    This example uses a small model to carry out a single vector matrix
    multiplication to demonstrate building and running a Keras model
    with build_model().

    This example will help identify what you should expect from each build_model()
    Keras build. You can find the build results in the cache directory at
    ~/.cache/turnkey_test_cache/hello_keras_world/ (unless otherwise specified).
"""

import tensorflow as tf
from turnkeyml import build_model

tf.random.set_seed(0)


# Define model class
class SmallKerasModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, output_size):
        super(SmallKerasModel, self).__init__()
        self.dense = tf.keras.layers.Dense(output_size, activation="relu")

    def call(self, x):  # pylint: disable=arguments-differ
        output = self.dense(x)
        return output


# Instantiate model and generate inputs
batch_size = 1
input_size = 10
output_size = 5
keras_model = SmallKerasModel(output_size)
keras_model.build(input_shape=(batch_size, input_size))
inputs = {"x": tf.random.uniform((batch_size, input_size), dtype=tf.float32)}

# Build the model
state = build_model(
    keras_model,
    inputs,
    build_name="hello_keras_world",
    cache_dir="~/.cache/turnkey_test_cache",
)

# Print build results
print(f"Build status: {state.build_status}")
