import tensorflow as tf

from dataset.util import apply_trans
from models.util import gram_schmidt
keras = tf.keras
layers = keras.layers

class VAE(keras.Model):
	def __init__(self, enc, dec, beta, J_tree, batch_rank):
		super().__init__(name='VAE')
		self.encoder = enc
		self.decoder = dec
		self.beta = beta
		self.J_tree = J_tree
		self.batch_rank = batch_rank

	def call(self, inputs):
		x = inputs
		# Apply forward kinematics

		# Flat of x[:, :3, :]
		x = tf.transpose(tf.transpose(x)[:, :3])

		z_mean, z_log_var = x = self.encoder(x)
		x = Sampling()(x)
		x = self.decoder(x)

		# Make it SO(3)
		x = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [22, 3, 2]], axis=-1))
		x = gram_schmidt(x)

		kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
		kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
		self.add_loss(kl_loss)
		self.add_metric(kl_loss, name="kl_loss")

		return x

class Sampling(layers.Layer):
	"""Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

	def call(self, inputs):
		z_mean, z_log_var = inputs
		epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
		return z_mean + tf.exp(0.5 * z_log_var) * epsilon