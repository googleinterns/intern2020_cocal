import os, sys

import tensorflow as tf

"""
precise confidence predictor
"""
class ConfPred(tf.keras.Model):
    def __init__(self, mdl_pred, mdl_w=None):
        super().__init__()
        self.mdl_pred = mdl_pred
        self.model_iw = mdl_w            
        self.T = tf.Variable(1.0)


    def call(self, x, training=False):
        out = self.mdl_pred(x, training=False)
        c = tf.math.reduce_max(tf.nn.softmax(out['logits']), axis=-1)
        if self.model_iw is None:
            w = tf.ones_like(c)
        else:
            w = self.model_iw(x)

        conf_bnd = c
        conf_iw = 1.0 - 0.5*tf.math.abs(w/(1+w) - 0.5)
        
        conf_adapt = tf.math.multiply(conf_bnd, conf_iw)
        conf_adapt = tf.cast(conf_adapt >= self.T, tf.int64)
        return conf_adapt


class NaiveConfPred(tf.keras.Model):
    def __init__(self, mdl_pred):
        super().__init__()
        self.mdl_pred = mdl_pred
        for l in self.mdl_pred.layers:
            l.trainable = False
        self.T = tf.Variable(1.0)


    def __call__(self, x, training=False):
        conf = tf.math.reduce_max(tf.nn.softmax(self.mdl_pred(x, training=False)['logits']), axis=-1)
        conf = tf.cast(conf >= self.T, tf.int64)
        return conf


class TwoParamsConfPred(tf.keras.Model):
    def __init__(self, mdl_pred, mdl_w=None):
        super().__init__()
        self.mdl_pred = mdl_pred
        self.model_iw = mdl_w            
        self.T_bnd = tf.Variable(1.0)
        self.T_iw = tf.Variable(1.0)


    def call(self, x, training=False):
        out = self.mdl_pred(x, training=False)
        c = tf.math.reduce_max(tf.nn.softmax(out['logits']), axis=-1)
        if self.model_iw is None:
            w = tf.ones_like(c)
        else:
            w = self.model_iw(x)

        conf_bnd = c
        conf_iw = 1.0 - 0.5*tf.math.abs(w/(1+w) - 0.5)

        conf_adapt = tf.cast((conf_bnd >= self.T_bnd) & (conf_iw >= self.T_iw), tf.int64)
        return conf_adapt
