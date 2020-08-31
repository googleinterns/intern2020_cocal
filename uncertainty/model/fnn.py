import os, sys

import tensorflow as tf
import numpy as np

from model.util import *


class SmallAdvFNN(tf.keras.Model):
    def __init__(self, n_in):
        super().__init__()

        self.model = tf.keras.Sequential([
            GradReverseLayer(),
            tf.keras.layers.Dense(1, input_shape=(n_in,)),
            tf.keras.layers.Activation('sigmoid')
        ], name='adv_net_classifier')

        
    def call(self, x, training=False):
        return self.model(x, training=training)
    

class MidAdvFNN(tf.keras.Model):
    def __init__(self, n_in, n_hiddens=500):
        super().__init__()

        self.model = tf.keras.Sequential([
            GradReverseLayer(),
            tf.keras.layers.Dense(n_hiddens, input_shape=(n_in,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_hiddens, input_shape=(n_hiddens,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),            
            tf.keras.layers.Dense(1, input_shape=(n_hiddens,)),
            tf.keras.layers.Activation('sigmoid')
        ], name='adv_net_classifier')

        
    def call(self, x, training=False):
        return self.model(x, training=training)            
            
            
class BigAdvFNN(tf.keras.Model):
    def __init__(self, n_in, n_hiddens=500):
        super().__init__()

        self.model = tf.keras.Sequential([
            GradReverseLayer(),
            tf.keras.layers.Dense(n_hiddens, input_shape=(n_in,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_hiddens, input_shape=(n_hiddens,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_hiddens, input_shape=(n_hiddens,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(n_hiddens, input_shape=(n_hiddens,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),            
            tf.keras.layers.Dense(1, input_shape=(n_hiddens,)),
            tf.keras.layers.Activation('sigmoid')
        ], name='adv_net_classifier')

        
    def call(self, x, training=False):
        return self.model(x, training=training)
                        
class FNN(tf.keras.Model):
    def __init__(self, n_in, n_out, n_hiddens=500, n_layers=4):
        super().__init__()
        
        self.model = tf.keras.Sequential()
        for i in range(n_layers):
            n = n_in if i == 0 else n_hiddens
            self.model.add(tf.keras.layers.Dense(n_hiddens, input_shape=(n,)))
            self.model.add(tf.keras.layers.ReLU())
            self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(n_out, input_shape=(n_hiddens if n_hiddens is not None else n_in,)))
        
        
    def call(self, x, training=False):
        logits = self.model(x, training=training)
        if logits.shape[1] == 1:
            probs = tf.math.sigmoid(logits)
        else:
            probs = tf.nn.softmax(logits, -1)
        return {'logits': logits, 'probs': probs, 'embeds': None}
            
    def train(self):
        pass

    def eval(self):
        pass


class Linear(FNN):
    def __init__(self, n_in, n_out, n_hiddens=None):
        super().__init__(n_in, n_out, n_hiddens=None, n_layers=0)


class SmallFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=1)

    
class MidFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=2)

        
class BigFNN(FNN):
    def __init__(self, n_in, n_out, n_hiddens=500):
        super().__init__(n_in, n_out, n_hiddens, n_layers=4)
    
    
    
