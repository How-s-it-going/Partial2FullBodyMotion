import tensorflow as tf
import tensorflow.keras as keras

from models.block import LinerResBlock
layers = keras.layers

class Decoder(keras.Model):
	def __init__(self, res_num, space_dim=256, activation=None, **kwargs):
		super().__init__(name='Decoder')
		self.activation = activation or layers.LeakyReLU()
		self.pre = layers.Dense(space_dim, activation=self.activation)
		self.resblocks = [LinerResBlock(space_dim, self.activation, **kwargs) for _ in range(res_num)]
		self.post = layers.Dense(22*6)

	def call(self, z):
		z = self.pre(z)
		for b in self.resblocks:
			z = b(z)
		z = self.activation(z)
		z = self.post(z)
		return z
