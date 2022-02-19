import tensorflow as tf
import tensorflow.keras as keras
layers = keras.layers

class LinearResBlock(keras.Model):
	def __init__(self, units=256, activation=None, normalization=None, **kwargs):
		super().__init__(name='LinearResBlock')
		self.b = ResBlock(lambda: layers.Dense(units, **kwargs), activation, normalization)

	def call(self, input):
		return self.b(input)

class ConvResBlock(keras.Model):
	def __init__(self, filters=256, kernel_size=3, activation=None, normalization=None, **kwargs):
		super().__init__(name='ConvResBlock')
		self.b = ResBlock(lambda: layers.Conv1D(filters, kernel_size, **kwargs), activation, normalization)

	def call(self, input):
		return self.b(input)

class ResBlock(keras.Model):
	def __init__(self, layer, activation=None, normalization=None):
		def f():
			return normalization() if normalization else layers.LayerNormalization()
		self.activation = activation or layers.LeakyReLU()
		super().__init__(name='ResBlock')
		self.norm = [f() for _ in range(2)]
		self.dense = [layer() for _ in range(2)]

	def call(self, input):
		x = input
		for n, d in zip(self.norm, self.dense):
			x = n(x)
			x = self.activation(x)
			x = d(x)

		return x + input
