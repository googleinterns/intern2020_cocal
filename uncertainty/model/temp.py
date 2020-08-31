import os, sys
import tensorflow as tf

class TempCls(tf.keras.Model):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.T = tf.Variable(1.0)
        self.dim_feat = self.model.dim_feat

        
    def __call__(self, x, training=False):
        out = self.model(x, training=False)
        logits = out['logits']
        logits = logits / self.T
        if logits.shape[1] == 1:
            probs = tf.math.sigmoid(logits)
            y_pred = tf.cast(probs > 0.5, tf.int64)

        else:
            probs = tf.nn.softmax(logits, -1)
            y_pred = tf.math.argmax(logits, -1)


        return {'logits': logits, 'probs': probs, 'y_pred': y_pred, 'embeds': out['embeds']}
    

    def train(self):
        for l in self.model.layers:
            l.trainable = False

    
