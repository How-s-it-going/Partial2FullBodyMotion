import tensorflow as tf
import tensorflow.keras as keras

from models.block import ConvResBlock, LinearResBlock

layers = keras.layers

class SequenceEncoder(keras.Model):
	def __init__(self, flames, res_num, d, space_dim=256, activation=None, **kwargs):
		super().__init__(name='SequenceEncoder')
		self.activation = activation or layers.LeakyReLU()
		self.lin = [layers.Dense(128, self.activation, ) for _ in range(flames)]
		self.cores = [ConvResBlock(space_dim, activation=self.activation) for _ in range(res_num)]
		self.lires = [LinearResBlock(space_dim, activation=self.activation) for _ in range(res_num)]
		self.norm = layers.LayerNormalization()
		self.mu, self.sig = layers.Dense(d), layers.Dense(d)

	def call(self, input):
		x = tf.stack([l(input[:, i]) for i, l in enumerate(self.lin)], -2)

		for l in self.cores:
			x = l(x)
		x = self.activation(x)

		x = layers.Flatten(x)
		x = self.norm(x)
		
		for l in self.lires:
			x = l(x)
		x = self.activation(x)

		return self.mu(x), self.sig(x)
