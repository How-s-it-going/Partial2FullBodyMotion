from functools import reduce
import os
import tensorflow as tf
import numpy as np
import glob

from dataset.util import *

# class AMASS_legacy:
# 	def __init__(self, ds: tf.data.Dataset):
# 		self.Dataset = ds

# 	def get_fullbody(self):
# 		# force it to face to certain direction
# 		return self.Dataset.flat_map(lambda x: x)

# 	def get_hh(self):
# 		# force it to face to certain direction
# 		return tf.data.Datset.from_tensor_slices(
# 			(nest_map(self.fullbody_to_hh, 1)(self.Dataset), self.Dataset))

# 	def get_hh_static(self):
# 		return self.get_hh().flat_map(lambda x: x)
		
# 	def get_hh_sequence(self, size):
# 		# TODO: follow paper to make both train and test data
# 		return self.get_hh().flat_map(
# 			lambda x: tf.data.Dataset(x).window(size, 4, drop_remainder=True))

# 	@staticmethod
# 	def fullbody_to_hh(pose: tf.Tensor):
# 		# TODO: impl
# 		return pose

# 	@staticmethod
# 	def load_datasets(
# 		amass_path: str, model_path: str, *, 
# 		whitelist: list[str]=None, 
# 		framerate_adjust: float=None):
# 		# Captures[Pose(J*3)]

# 		if model_path is None:
# 			raise ValueError("model_path must be specified")
# 		model = np.load(model_path)

# 		dirs = os.listdir(amass_path)
# 		if whitelist is not None:
# 			dirs = [os.path.join(amass_path, i) for i in dirs if i in whitelist]
# 		ds = tf.data.Dataset.from_tensor_slices(dirs)
# 		# DS[Captures[Pose(J*3)]]

# 		def load_dataset(ds: str):
# 			def load_npz(ds):
# 				def inner(path):
# 					t = np.load(path)
# 					poses = t['poses'][::max(1, round(t['mocap_framerate']/framerate_adjust)) if framerate_adjust is not None else 1, :66]
					
# 				ds = tf.data.Dataset.from_tensor_slices(tf.py_function(inner, [ds], tf.float32))
# 				ds = ds.map(lambda x: tf.reshape(x, [-1, 3]))
# 				# VTM
# 				ds = ds.map(VTM)
# 				# To SE(3)
# 				return ds
# 			ds = tf.data.Dataset.list_files(f'{ds}/**/+_poses.npz')
# 			ds = ds.map(load_npz)
# 			return ds

# 		ds = ds.flat_map(load_dataset)
# 		return AMASS(ds)

class AMASS:
	def __init__(self, amass_path: str, model_path: str, 
		whitelist: list[str]=None, framerate_adjust: float=None):
		self.amass_path = amass_path
		self.whitelist = whitelist
		self.model_path = model_path
		self.dirs = os.listdir(self.amass_path)
		if self.whitelist is not None:
			self.dirs = [os.path.join(self.amass_path, i) for i in self.dirs if i in self.whitelist]
		self.ds = [load_file(c, self.model_path) for c in self.dirs]
	
	def get_fullbody(self, do_framerate_adjust=False):
		def so(a, b):
			t = TSO(tf.reshape(b[:66], (-1, 3)))
			t = t @ get_force_rot(t)
			return a, t

		ds = reduce(lambda a, b: a.concatenate(b), self.ds)
		if do_framerate_adjust and self.framerate_adjust:
			ds = ds.map(lambda a, b: adjust_framerate(a, b, self.framerate_adjust))
		ds = ds.map(lambda a, b: (a.map(so), b))
		ds = ds.flat_map(lambda a, b: a.map(lambda i, j: (i, TSE(j, b['joint']))))

		return ds