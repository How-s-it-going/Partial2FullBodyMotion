import glob
import random
import tensorflow as tf
import numpy as np

def nest_map(func, depth, num_parallel_calls=None, deterministic=None):
	def inner(ds: tf.data.Dataset):
		return ds.map(func, num_parallel_calls, deterministic)
	return inner if depth == 0 else nest_map(inner, depth-1, num_parallel_calls, deterministic)

@tf.function
def apply_trans(trans, J_tree, batch_rank=0):
	# return tf.concat([trans[:1], trans[1:] @ tf.gather_nd(trans, tf.expand_dims(J_tree[0,1:22], axis=-1))], axis=-3)
	if batch_rank != 0:
		perm = tf.range(tf.rank(trans))
		perm = tf.tensor_scatter_nd_update(perm, [[0], [batch_rank]], [batch_rank, 0])
		trans = tf.transpose(trans, perm)

	for i in range(1, 22):
		trans = tf.tensor_scatter_nd_update(trans, [[i]], [trans[i] @ trans[J_tree[0, i]]])

	if batch_rank != 0:
		trans = tf.transpose(trans, perm)
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
	A = tf.stack([zero, -z, y], axis=-1)
	B = tf.stack([z, zero, -x], axis=-1)
	C = tf.stack([-y, x, zero], axis=-1)
	K = tf.stack([A, B, C], axis=-1)
	I = tf.eye(3)
	return I + K*tf.expand_dims(tf.sin(angle), axis=-1) \
		+ (K @ K)*(1 - tf.expand_dims(tf.cos(angle), axis=-1))

@tf.function
def TSE(rot, trans):
	tpad = tf.scatter_nd([[1, 1]], [1], [3,2])
	rpad = tf.concat([tf.zeros((tf.rank(rot)-3, 2), tf.int32), tpad], 0)
	trans = tf.pad(tf.expand_dims(trans, axis=-1), tpad, constant_values=1)
	trans = tf.reshape(trans, tf.concat([tf.ones([tf.rank(rot)-3], tf.int32), [-1, 4, 1]], 0))
	trans = tf.tile(trans, tf.concat([tf.shape(rot)[:-3], tf.ones([3], tf.int32)], 0))
	return tf.concat([tf.pad(rot, rpad), trans], axis=-1)

@tf.function
def vertices2joints(regressor, vertices):
	return tf.einsum('bik,ji->bjk', vertices, regressor)

@tf.function
def blend_shapes(betas, shape_dirs):
	return tf.einsum('bl,mkl->bmk', betas, shape_dirs)

@tf.function
def get_base_joint(v_template, regressor, shape_dirs, betas):
	return tf.squeeze(vertices2joints(regressor, 
		v_template + blend_shapes(betas if tf.rank(betas) > 1 else tf.expand_dims(betas, 0), shape_dirs)))

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

	j = [get_base_joint(model['v_template'].astype(np.float32), model['J_regressor'].astype(np.float32), 
		model['shapedirs'].astype(np.float32), n['betas'].astype(np.float32)) for n in npz]
	j = [tf.concat([i[0:1], i[1:22] - tf.gather_nd(i, tf.expand_dims(model['kintree_table'][0, 1:22], axis=-1))], axis=-2) for i in j]	

	dic = tf.data.Dataset.from_tensor_slices(
		{'joint': j, 'framerate': [n['mocap_framerate'].astype(np.float32) for n in npz]})
	return tf.data.Dataset.zip((ds, dic))

def load_file(dir, model):
	ds = tf.data.Dataset.list_files(f'{dir}/**/*_poses.npz', shuffle=False)
	ds = ds.shuffle(1000, reshuffle_each_iteration=False)
	# file_list = glob.glob(f'{dir}/**/*_poses.npz', recursive=True)
	# random.shuffle(file_list)
	# ds = tf.data.Dataset.from_tensor_slices(file_list)
	# ds = ds.apply(tf.data.experimental.assert_cardinality(len(file_list)))

	reg = model['J_regressor'].astype(np.float32)
	sha = model['shapedirs'].astype(np.float32)
	temp = model['v_template'].astype(np.float32)
	table = tf.expand_dims(model['kintree_table'][0, 1:22], axis=-1)

	def inner(f):
		def load_npz(i):
			n = np.load(i)
			return (
				# n['trans'].astype(np.float32),
				n['poses'].astype(np.float32),
				n['betas'].astype(np.float32),
				n['mocap_framerate'].astype(np.float32),
				n['trans'].shape[0])

		# t, p, b, m, f = tf.numpy_function(load_npz, [f], [tf.float32, tf.float32, tf.float32, tf.float32, tf.int32])
		p, b, m, f = tf.numpy_function(load_npz, [f], [tf.float32, tf.float32, tf.float32, tf.int32])
		
		ds =tf.data.Dataset.from_tensor_slices(p)
		j = get_base_joint(temp, reg, sha, b)
		j = tf.concat([j[0:1], j[1:22] - tf.gather_nd(j, table)], axis=-2)
		dic = {'joint': j, 'framerate': m, 'frames': f}
		return (ds, dic)

	return ds.map(inner)

@tf.function
def adjust_framerate(ds, dic, framerate):
	f = dic['framerate']
	di = dic.copy()
	step = tf.maximum(tf.constant(1, tf.int64), tf.cast(tf.round(f/framerate), tf.int64))
	di['frames'] = tf.math.ceil(tf.cast(dic['frames'], tf.int64)/step)
	return ds.shard(step, 0), di