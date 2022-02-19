import tensorflow as tf
import tensorflow.keras as keras
layers = keras.layers

class LinearResBlock(keras.Model):
	def __init__(self, units=256, activation=None, normalization=None, **kwargs):
		def f():
			return normalization() if normalization else layers.LayerNormalization()
		self.activation = activation or layers.LeakyReLU()
		super().__init__(name='LinerResBlock')
		self.norm = [f() for _ in range(2)]
		self.dense = [layers.Dense(units, activation, **kwargs) for _ in range(2)]

	def call(self, input):
		x = input
		for n, d in zip(self.norm, self.dense):
			x = n(x)
			x = self.activation(x)
			x = d(x)

		return x + input