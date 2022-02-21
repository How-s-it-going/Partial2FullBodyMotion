import tensorflow as tf
import tensorflow.keras as keras

from dataset.util import TSE, apply_trans
layers = keras.layers

def gram_schmidt(rot):
	a, b = tf.unstack(rot, axis=-2)
	a = a/tf.norm(a, axis=-1)
	b = b - a*mm(b, a, keepdims=True)
	b = b/tf.norm(b, axis=-1)
	c = tf.linalg.cross(a, b)
	return tf.stack([a, b, c], axis=-2)

def build_reconstruction_error(J_tree):
	def reconstruction_error(y_true, y_pred):
		sigmat = 0.02
		sigmar = 0.1

		y_true = apply_trans(y_true, J_tree, 1)
		t_t, t_r = get_trans(y_true), get_rot(y_true)

		# TO SE(3) and apply forward kinematics
		y_pred = TSE(y_pred, t_t)
		y_pred = apply_trans(y_pred, J_tree, 1)
		p_t, p_r = get_trans(y_pred), get_rot(y_pred)

		dt = p_t - t_t

		perm = tf.range(tf.rank(t_r)-1, -1, -1)
		perm = tf.tensor_scatter_nd_update(perm, [tf.shape(t_r)-1, tf.shape(t_r)-2], [1, 0])
		dr = get_so_lie_alg(tf.transpose(t_r, perm)@p_r)

		loss = tf.reduce_mean(sigmat*mm(dt, dt) + sigmar*mm(dr, dr), axis=-1)
		return loss
	
	return reconstruction_error

def mm(a, b, keepdims=False):
	return tf.reduce_sum(a*b, axis=-1, keepdims=keepdims)

def get_trans(mat):
	# [:, :3, 3]
	return tf.transpose(tf.transpose(mat)[3, :3])

def get_rot(mat):
	# [:, :3, :3]
	return tf.transpose(tf.transpose(mat)[:3, :3])

def get_so_lie_alg(so):
	perm = tf.range(tf.rank(so)-1, -1, -1)
	perm = tf.tensor_scatter_nd_update(perm, [tf.shape(perm)-1, tf.shape(perm)-2], [1, 0])
	A = (so - tf.transpose(so, perm)) /2
	# Trace(A^2) always be negative since it's skew-symmetric
	detA = tf.sqrt(-tf.linalg.trace(A@A)/2)
	r = (tf.experimental.numpy.arcsin(detA)/detA)*A
	r1 = tf.gather(r, 0, axis=-1)
	r2 = tf.gather(r, 1, axis=-1)
	x = tf.gather(r1, 1, axis=-1)
	y = tf.gather(r1, 2, axis=-1)
	z = tf.gather(r2, 0, axis=-1)
	return tf.stack([x, y, z], axis=-1)
