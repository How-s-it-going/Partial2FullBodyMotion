from functools import reduce
import os
import tensorflow as tf
import numpy as np
import glob

from dataset.util import *

def conbine(dss):
	return reduce(lambda a, b: (i.concatnate(j) for i, j in zip(a, b)), dss)

class AMASS:
	def __init__(self, amass_path: str, model_path: str,
		whitelist: list[str]=None, framerate_adjust: float=None):
		self.amass_path = amass_path
		self.whitelist = whitelist
		self.framerate_adjust = framerate_adjust
		self.model = np.load(model_path)
		self.dirs = os.listdir(self.amass_path)
		if self.whitelist is not None:
			self.dirs = [os.path.join(self.amass_path, i) for i in self.dirs if i in self.whitelist]
		self.ds = [load_file(c, self.model) for c in self.dirs]

	def get_fullbody(self, do_framerate_adjust=False):
		def so(f):
			t = TSO(tf.reshape(f[:66], (22, 3)))
			t = t @ get_force_rot(t)
			return t

		def inner(ds: tf.data.Dataset):
			# Cardinality will be lost after the filter is applied
			# Expecting that most of sequences contain more than 64 frames
			num = tf.cast(tf.round(ds.cardinality()/20), tf.int64)
			if self.framerate_adjust:
				ds = ds.map(lambda a, b: adjust_framerate(a, b, self.framerate_adjust))
			ds = ds.filter(lambda a, _: a.cardinality() >= 64)
			ds = ds.map(lambda a, b: (a.map(so), b))
			trl = ds.skip(num)
			tel = ds.take(num)
			tr, te = map(lambda d: d.flat_map(lambda a, b: a.map(lambda f: TSE(f, b['joint']))), (trl, tel))
			return (tr, tr, te, te)

		ds = [inner(d) for d in self.ds]
		(tr, trl, te, tel) = conbine(ds)
		tr, te = map(lambda d: d.map(lambda b: apply_trans(b, self.model['kintree_table'])), (tr, te))
		return (tr, trl), (te, tel)

	def get_hh_sequence(self):
		def so(f):
			t = TSO(tf.reshape(f[:, :66], (-1, 22, 3)))
			t = t @ get_force_rot(t[-1])
			return t

		def inner(ds: tf.data.Dataset):
			# Cardinality will be lost after the filter is applied
			# Expecting that most of sequences contain more than 64 frames
			num = tf.cast(tf.round(ds.cardinality()/20), tf.int64)
			if self.framerate_adjust:
				ds = ds.map(lambda a, b: adjust_framerate(a, b, self.framerate_adjust))
			ds = ds.filter(lambda _, b: b['frames'] >= 64)
			def t(num):
				return lambda a, b: (a.window(num, shift=4, drop_remainder=True).flat_map(lambda f: f.batch(num)), b)
			trl = ds.skip(num).map(t(16))
			tel = ds.take(num).map(t(64))
			trl, tel = map(lambda d: d.map(lambda a, b: (a.map(so), b)), (trl, tel))
			trl, tel = map(lambda d: d.flat_map(lambda a, b: a.map(lambda f: TSE(f, b['joint']))), (trl, tel))
			ds = map(lambda d: d.map(lambda b: apply_trans(b, self.model['kintree_table'], 1)), (trl, tel))
			tr, te = map(lambda d: d.map(lambda b: tf.gather(b, [15, 20, 21], axis=1)), ds)
			t = lambda fs: fs[-1]
			return (tr, trl.map(t), te, tel.map(t))

		ds = [inner(d) for d in self.ds]
		(tr, trl, te, tel) = conbine(ds)
		return (tr, trl), (te, tel)

	def get_hh(self, do_framerate_adjust=False):
		# [15, 20, 21] (head, right hand, left hand)
		(tr, trl), (te, tel) = self.get_fullbody(do_framerate_adjust)
		tr, te = map(lambda d: d.map(lambda b: tf.gather(b, [15, 20, 21])), (tr, te))
		return (tr, trl), (te, tel)
