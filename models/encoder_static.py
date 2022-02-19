import tensorflow as tf
from tensorflow.keras import Model, layers

from models.block import LinearResBlock


class StaticEncoder(Model):
    def __init__(self, res_num=3, d=30, activation=None, **kwargs):
        super().__init__(name='StaticEncoder')
        self.activation = activation or layers.LeakyReLU()
        self.pre = layers.Dense(256)
        self.resblocks = [LinearResBlock(256, self.activation, **kwargs) for _ in range(res_num)]
        self.post = layers.Dense(2 * d)

    def call(self, inputs, training=None, mask=None):
        x = self.pre(inputs)
        x = self.activation(x)
        for b in self.resblocks:
            x = b(x)
        x = self.activation(x)
        x = self.post(x)
        mu, sigma = tf.split(x, num_or_size_splits=2, axis=-1)

        return mu, sigma