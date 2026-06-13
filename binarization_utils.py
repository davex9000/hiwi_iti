"""
Minimal binary_dense layer compatible with the notebook API.

When BINARY=False (default in all notebooks): behaves as a standard Dense layer.
When BINARY=True: binarizes weights to {-1, +1} using a straight-through estimator
so that gradients still flow during training.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense


def binary_dense(n_in, n_out, levels=True, BINARY=False, LOGIC_SHRINKAGE=False,
                 custom_rand_seed=1, **kwargs):
    if BINARY:
        return _BinaryDenseLayer(units=n_out, seed=custom_rand_seed, **kwargs)
    return Dense(units=n_out, **kwargs)


class _BinaryDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, seed=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.seed = seed

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel",
            shape=(int(input_shape[-1]), self.units),
            initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs, training=None):
        # Straight-through estimator: binarise weights in the forward pass,
        # but let gradients pass through as if weights were real-valued.
        w_bin = self.w + tf.stop_gradient(tf.sign(self.w) - self.w)
        return tf.matmul(inputs, w_bin) + self.b

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "seed": self.seed})
        return config
