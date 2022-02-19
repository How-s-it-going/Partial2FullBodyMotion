import tensorflow as tf
from tensorflow.keras import Model, models, layers

from models.block import LinerResBlock


class StaticEncoder(Model):
    def __init__(self, joints=3, res_num=3, latent_dim=256, activation=None, **kwargs):
        super().__init__(name='Encoder_hh_Static')
        self.activation = activation or layers.LeakyReLU()
        self.input = layers.Input(shape=(joints * 12,), name='')
        self.pre = layers.Dense(256)
        self.resblocks = []