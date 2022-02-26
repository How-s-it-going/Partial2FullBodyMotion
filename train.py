import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp
import numpy as np
import json
import os
from urllib.parse import urlparse, urlunparse
import argparse

import dataset
import models


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='config.json', help='Path to config file.')
    parser.add_argument('--webhook_url', type=str, default='',
                        help='Discord webhook url for notification.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    args = parser.parse_args()

    return args


def m0(beta, z_dim, res_num, J_tree, activation=None):
    # 0. full body
    inp = keras.layers.Input((22, 4, 4))
    enc = models.StaticEncoder(res_num, z_dim, activation)
    dec = models.Decoder(res_num, activation=activation)
    vae = models.VAE(enc, dec, beta, J_tree, 1)
    return dec, vae


def m1(beta, z_dim, res_num, J_tree, activation=None):
    # 1. hh
    inp = keras.layers.Input((3, 4, 4))
    enc = models.StaticEncoder(res_num, z_dim, activation)
    dec = models.Decoder(res_num, activation=activation)
    vae = models.VAE(enc, dec, beta, J_tree, 1)
    return vae


def m2(beta, z_dim, res_num, J_tree, activation=None):
    # 1. hh
    inp = keras.layers.Input((16, 3, 4, 4))
    enc = models.SequenceEncoder(16, res_num, z_dim, activation=activation)
    dec = models.Decoder(res_num, activation=activation)
    vae = models.VAE(enc, dec, beta, J_tree, 1)
    return vae


def m3(beta, z_dim, res_num, J_tree, trained_dec, activation=None):
    # 1. hh
    inp = keras.layers.Input((3, 4, 4))
    enc = models.StaticEncoder(res_num, z_dim, activation)
    dec = trained_dec
    dec.trainable = False
    vae = models.VAE(enc, dec, beta, J_tree, 2)
    return vae


def m4(beta, z_dim, res_num, J_tree, trained_dec, activation=None):
    # 1. hh
    inp = keras.layers.Input((16, 3, 4, 4))
    enc = models.SequenceEncoder(16, res_num, z_dim, activation=activation)
    dec = trained_dec
    dec.trainable = False
    vae = models.VAE(enc, dec, beta, J_tree, 2)
    return vae


def _train(vae: keras.Model, tr, te, batch_size, epoch_num, callbacks=None):
    vae.compile(loss=models.build_reconstruction_error(J_tree))
    return vae.fit(
        tf.data.Dataset.zip(tr).batch(batch_size),
        validation_data=tf.data.Dataset.zip(te).batch(batch_size),
        epochs=epoch_num, callbacks=callbacks,
    )


def train(hparams, batch_size, epochs, callbacks=[]):
    def save_dir(
        s): return f'{hparams[HP_BETA]}_{hparams[HP_Z_DIMS]}_{hparams[HP_RES_NUM]}/{s}'
    callbacks = callbacks + [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(patience=20),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    def cb(s): return callbacks + [
        keras.callbacks.TensorBoard(
            'logs/fit/'+save_dir(s), histogram_freq=10, write_images=True),
        keras.callbacks.ModelCheckpoint(
            'saved_models/'+save_dir(s), save_best_only=True, save_weights_only=True),
        hp.KerasCallback('logs/hp_tuning/'+save_dir(s), hparams)
    ]

    tr, te = amass.get_fullbody()
    dec, v0 = m0(hparams[HP_BETA], hparams[HP_Z_DIMS],
                 hparams[HP_RES_NUM], J_tree)
    _train(v0, tr, te, batch_size, epochs, cb('full_body'))

    tr, te = amass.get_hh()
    v1 = m1(hparams[HP_BETA], hparams[HP_Z_DIMS], hparams[HP_RES_NUM], J_tree)
    _train(v1, tr, te, batch_size, epochs, cb('hh_static'))

    tr, te = amass.get_hh_sequence()
    v2 = m2(hparams[HP_BETA], hparams[HP_Z_DIMS], hparams[HP_RES_NUM], J_tree)
    _train(v2, tr, te, batch_size, epochs, cb('hh_sequence'))

    tr, te = amass.get_hh()
    v3 = m3(hparams[HP_BETA], hparams[HP_Z_DIMS],
            hparams[HP_RES_NUM], J_tree, dec)
    _train(v3, tr, te, batch_size, epochs, cb('hh_static_pre'))

    tr, te = amass.get_hh_sequence()
    v4 = m4(hparams[HP_BETA], hparams[HP_Z_DIMS],
            hparams[HP_RES_NUM], J_tree, dec)
    _train(v4, tr, te, batch_size, epochs, cb('hh_sequence'))


if __name__ == '__main__':
    args = get_opts()
    config = json.load(open(args.config_path))
    ds_config = config['dataset']
    hyper_params = config['hyper_params']
    HP_BETA = hp.HParam('beta', hp.Discrete(hyper_params['beta']))
    HP_Z_DIMS = hp.HParam('z_dims', hp.Discrete(hyper_params['z_dims']))
    HP_RES_NUM = hp.HParam('res_num', hp.Discrete(hyper_params['res_num']))

    amass = dataset.AMASS(
        ds_config['amass_path'],
        whitelist=ds_config['whitelist'],
        model_path=ds_config['model_filepath'],
        framerate_adjust=ds_config['framerate_adjust']
    )

    print('Loading body model...')
    model = np.load(ds_config['model_filepath'])
    J_tree = model['kintree_table']

    callbacks = []
    if args.webhook_url:
        url = urlparse(args.webhook_url)
        callbacks = callbacks + [keras.callbacks.RemoteMonitor(
            urlunparse(url._replace(path='')), url.path,
            field='content', send_as_json=False
        )]

    session_num = 0

    for beta in HP_BETA.domain.values:
        for z_dims in HP_Z_DIMS.domain.values:
            for res_num in HP_RES_NUM.domain.values:
                hparams = {
                    HP_BETA: beta,
                    HP_Z_DIMS: z_dims,
                    HP_RES_NUM: res_num,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                train(hparams, args.batch_size, args.num_epoch, callbacks)
                session_num += 1
