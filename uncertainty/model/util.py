import os, sys
import tensorflow as tf

class IW(tf.keras.Model):
    def __init__(self, model_sd, iw_max=1e6):
        super().__init__()
        self.model_sd = model_sd
        self.iw_max = iw_max

    def __call__(self, x, training=False, iw=True):
        probs = self.model_sd(x, training=False)['probs']
        g = probs[:, 1]
        if iw:
            return tf.math.minimum(1/g - 1.0, self.iw_max)
        else:
            return g

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


class GradReverseLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        
    def call(self, x, training=False):
        if training:
            return grad_reverse(x)
        else:
            return x


class DAN(tf.keras.Model):
    def __init__(self, model, model_adv):
        super().__init__()
        self.model = model
        self.model_adv = model_adv

    def call(self, x, training=False):
        out = self.model(x, training=training)
        out['domain'] = self.model_adv(out['embeds'], training=training)
        return out



def set_trainable(model, mode):
    for l in model.layers:
        l.trainable = mode
