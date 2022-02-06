import glob
import tensorflow as tf
import numpy as np

def nest_map(func, depth, num_parallel_calls=None, deterministic=None):
	def inner(ds: tf.data.Dataset):
		return ds.map(func, num_parallel_calls, deterministic)
	return inner if depth == 0 else nest_map(inner, depth-1, num_parallel_calls, deterministic)

@tf.function
def apply_trans(trans, J_tree):
	# return tf.concat([trans[:1], trans[1:] @ tf.gather_nd(trans, tf.expand_dims(J_tree[0,1:22], axis=-1))], axis=-3)
	for i in range(1, 22):
		trans = tf.tensor_scatter_nd_update(trans, [[i]], [trans[i] @ trans[J_tree[0, i]]])
	return trans
	
@tf.function
def fit_shape_to(tensor, shape):
	s = tensor.shape
	tensor = tf.reshape(tensor, [1]*len(shape)+list(s))
	tensor = tf.tile(tensor, list(shape)+[1]*len(s))
	return tensor

@tf.function
def TSO(rot_vectors: np.ndarray):
	angle = tf.norm(rot_vectors, axis=-1, keepdims=True)
	vec = rot_vectors / (angle + 1e-10)
	x, y, z = tf.unstack(vec, axis=-1)
	zero = tf.zeros_like(x)
	K = tf.stack([zero, -z, y,  z, zero, -x,  -y, x, zero], axis=-1)
	K = tf.reshape(K, [-1, 3, 3])
	I = tf.eye(3)
	return I + K*tf.expand_dims(tf.sin(angle), axis=-1) \
		+ (K @ K)*(1 - tf.expand_dims(tf.cos(angle), axis=-1))

@tf.function
def TSE(rot, trans):
	pad = tf.tile([[0, 0]], [tf.rank(rot), 1])
	pad = pad + tf.scatter_nd([[tf.rank(rot)-2, 1]], [1], pad.shape)
	return tf.concat([tf.pad(rot, pad), tf.pad(tf.expand_dims(trans, axis=-1), pad, constant_values=1)], axis=-1)

@tf.function
def vertices2joints(regressor, vertices):
	return tf.einsum('bik,ji->bjk', vertices, regressor)

@tf.function
def blend_shapes(betas, shape_dirs):
	return tf.einsum('bl,mkl->bmk', betas, shape_dirs)

@tf.function
def get_base_joint(regressor, shape_dirs, betas):
	return tf.squeeze(vertices2joints(regressor, 
		blend_shapes(betas if tf.rank(betas) > 1 else tf.expand_dims(betas, 0), shape_dirs)))

@tf.function
def get_force_rot(rot: tf.Tensor):
	angle = tf.atan2(-rot[:,1,0], rot[:,0,0])
	s = tf.sin(angle)
	c = tf.cos(angle)
	R = tf.stack([c, -s, s, c], axis=-1)
	R = tf.reshape(R, [-1, 2, 2])
	R = tf.pad(R, [[0,0],[0,1],[0,1]])
	return R + tf.scatter_nd([[2,2]], [1.], [3,3])

def _load_file(dir, model_path):
	ds = glob.glob(f'{dir}/**/*_poses.npz', recursive=True)
	npz = [dict(np.load(f)) for f in ds]
	model = np.load(model_path)
	ds = tf.data.Dataset.from_tensor_slices(
		[tf.data.Dataset.from_tensor_slices(
			(n['trans'].astype(np.float32), n['poses'].astype(np.float32))) for n in npz])

	j = [get_base_joint(model['J_regressor'].astype(np.float32), 
		model['shapedirs'].astype(np.float32), n['betas'].astype(np.float32)) for n in npz]
	j = [tf.concat([i[0:1], i[1:22] - tf.gather_nd(i, tf.expand_dims(model['kintree_table'][0, 1:22], axis=-1))], axis=-2) for i in j]	

	dic = tf.data.Dataset.from_tensor_slices(
		{'joint': j, 'framerate': [n['mocap_framerate'].astype(np.float32) for n in npz]})
	return tf.data.Dataset.zip((ds, dic))

def load_file(dir, model):
	ds = tf.data.Dataset.list_files(f'{dir}/**/*_poses.npz')
	reg = model['J_regressor'].astype(np.float32)
	sha = model['shapedirs'].astype(np.float32)
	table = tf.expand_dims(model['kintree_table'][0, 1:22], axis=-1)

	def inner(f):
		def load_npz(i):
			n = np.load(i)
			return (
				n['trans'].astype(np.float32), 
				n['poses'].astype(np.float32), 
				n['betas'].astype(np.float32),
				n['mocap_framerate'].astype(np.float32))

		t, p, b, m = tf.numpy_function(load_npz, [f], [tf.float32, tf.float32, tf.float32, tf.float32])
		
		ds =tf.data.Dataset.from_tensor_slices((t, p))
		j = get_base_joint(reg, sha, b)
		j = tf.concat([j[0:1], j[1:22] - tf.gather_nd(j, table)], axis=-2)
		dic = {'joint': j, 'framerate': m}
		return (ds, dic)

	return ds.map(inner)

@tf.function
def adjust_framerate(ds, dic, framerate):
	f = dic['framerate']
	step = tf.maximum(tf.constant(1, tf.int64), tf.cast(tf.round(f/framerate), tf.int64))
	return ds.shard(step, 0), dic