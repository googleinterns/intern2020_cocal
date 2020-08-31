import os, sys

import tensorflow as tf

from model.util import *

        
class SmallCNN(tf.keras.Model):
    def __init__(self, num_class, input_shape, activation=None):
        super().__init__()
        self.feat = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=20, kernel_size=5),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=50, kernel_size=5),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(500, input_shape=(800,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.5),
        ], name='small_cnn_feat')

        self.dim_feat = 500

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(num_class, input_shape=(500,)),
        ], name='small_cnn_classifier')
            
        
    def call(self, x, training=False):
        z = self.feat(x, training=training)
        logits = self.classifier(z, training=training)
        y_pred = tf.math.argmax(logits, -1)
        return {'embeds': z, 'logits': logits, 'y_pred': y_pred}
            
            
