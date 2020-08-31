import os, sys

import tensorflow as tf
import tensorflow_probability as tfp

import model

class SourceDisc(tf.keras.Model):
    def __init__(self, model_bb, model_head):
        super().__init__()
        self.model_bb = model_bb
        self.model_head = model_head
        self.dim_feat = self.model_bb.dim_feat

    def call(self, x, training=False):
        feat = self.model_bb(x, training=False)['embeds']
        return self.model_head(feat, training=training)

    
    def train(self):
        for l in self.model_bb.layers:
            l.trainable = False
        for l in self.model_head.layers:
            l.trainable = True

            

class SourceDisc_old(tf.keras.Model):
    def __init__(self, mdl=None, model_name='ResNet18', activation='relu', input_shape=(32, 32, 3)):
        super().__init__()
        if mdl is None:
            self.use_feat = False
            self.mdl = getattr(model, model_name)(num_class=2, activation=activation, input_shape=input_shape)
        else:
            self.use_feat = True
            self.mdl = mdl
            for l in self.mdl.layers:
                l.trainable = False
            self.classifier = tf.keras.layers.Dense(2)

        
    def __call__(self, x, training=False):
        if self.use_feat:
            feat = self.mdl(x, training=False)['embeds']
            logits = self.classifier(feat)
        else:
            logits = self.mdl(x, training=training)['logits']
        return {'logits': logits}


class OneHiddenSourceDisc(SourceDisc):
    def __init__(self, mdl=None, model_name='ResNet18', activation='relu', input_shape=(32, 32, 3)):
        super().__init__(mdl, model_name, activation, input_shape)
        self.classifier = tf.keras.Sequential(
            [   
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(2)
            ]
        )
            
    

def k_gaussian(x, y, sigma):
    """
    x = n x d
    y = n x d
    """
    x = tf.expand_dims(x, 1)
    y = tf.expand_dims(y, 0)
    k = tf.math.exp(- tf.math.reduce_sum(tf.math.pow(x - y, 2.0), -1) / tf.math.pow(sigma, 2.0))
    return k

class GaussianBiasedMMD(tf.keras.Model):
    def __init__(self, mdl, ld_tar, sigma=1.0):
        super().__init__()
        self.mdl = mdl
        self.ld_tar = ld_tar
        self.sigma = sigma

        for l in self.mdl.layers:
            l.trainable = False


    def __call__(self, x, training=False):
        feat = self.mdl(x, training=False)['embeds']
        ks = []
        for x_tar, _ in self.ld_tar:
            feat_tar = self.mdl(x_tar, training=False)['embeds']
            ks.append(
                k_gaussian(
                    tf.reshape(feat, (feat.shape[0], -1)),
                    tf.reshape(feat_tar, (feat_tar.shape[0], -1)),
                    self.sigma
                )
            )
        ks = tf.concat(ks, -1)
        k_sum = tf.math.reduce_sum(ks, 1)
        n = ks.shape[1]

        ## compute biased mmd
        mmd_b = tf.math.sqrt(
            k_sum/n**2 - 2*k_sum/n + k_sum
        )
        #mmd_b = tf.stop_gradient(mmd_b).numpy()
        return mmd_b + 1e-16
            
            
        
class IW_MoG(tf.keras.Model):
    def __init__(self, p, q, model_conf=None):
        super().__init__()
        self.p = p
        self.q = q
        self.model_conf = model_conf
        

    def _compute_p_q_conf(self, model_conf, x_rng, y_rng, inc):
        x = tf.range(x_rng[0], x_rng[1]+inc, inc)
        y = tf.range(y_rng[0], y_rng[1]+inc, inc)
        X, Y = tf.meshgrid(x, y)
        pnts = tf.concat((tf.reshape(X, (-1, 1)), tf.reshape(Y, (-1, 1))), 1)
        conf = model_conf(pnts)
        pnts_conf = pnts[conf==1]

        if len(pnts_conf) == 0:
            p_conf = 1e-16
            q_conf = 1e-16
        else:
            p_conf = tf.math.reduce_sum(self.p.prob(pnts_conf))
            q_conf = tf.math.reduce_sum(self.q.prob(pnts_conf))

        return p_conf, q_conf
    

    def _compute_cond_prob(self, x, p, p_cond, model_conf):
        conf_prob = tf.cast(model_conf(x), tf.float32)
        p = p.prob(x)
        cond_prob = conf_prob * p / p_cond
        return cond_prob

        
    def __call__(self, x, training=False):
        if self.model_conf:
            p_conf, q_conf = self._compute_p_q_conf(self.model_conf, x_rng=[-2.5, 2.5], y_rng=[-2.5, 2.5], inc=0.005)
            p_prob = self._compute_cond_prob(x, self.p, p_conf, self.model_conf)
            q_prob = self._compute_cond_prob(x, self.q, q_conf, self.model_conf)
        else:
            p_prob = self.p.prob(x)
            q_prob = self.q.prob(x)
        w = q_prob / (p_prob + 1e-16)
        return w

        

if __name__ == '__main__':
    from data.util import *

    rot_deg = 60

    mu_pos = [+0.6, +0.6]
    mu_neg = [-0.6, -0.6]
    
    cov_pos = np.diag([0.01, 0.1])
    cov_neg = np.diag([0.01, 0.1])
    
    mu_pos_rot, cov_pos_rot = rot_gaussian(rot_deg, mu_pos, cov_pos)
    mu_neg_rot, cov_neg_rot = rot_gaussian(rot_deg, mu_neg, cov_neg)

    mu_pos, mu_neg = tf.constant(mu_pos, dtype=tf.float32), tf.constant(mu_neg, dtype=tf.float32)
    cov_pos, cov_neg = tf.constant(cov_pos, dtype=tf.float32), tf.constant(cov_neg, dtype=tf.float32)
    mu_pos_rot, mu_neg_rot = tf.constant(mu_pos_rot, dtype=tf.float32), tf.constant(mu_neg_rot, dtype=tf.float32)
    cov_pos_rot, cov_neg_rot = tf.constant(cov_pos_rot, dtype=tf.float32), tf.constant(cov_neg_rot, dtype=tf.float32)
    
    
    p_pos = tfp.distributions.MultivariateNormalTriL(loc=mu_pos, scale_tril=tf.linalg.cholesky(cov_pos))
    p_neg = tfp.distributions.MultivariateNormalTriL(loc=mu_neg, scale_tril=tf.linalg.cholesky(cov_neg))
    #p_pos = tfp.distributions.MultivariateNormalFullCovariance(mu_pos, cov_pos)
    #p_neg = tfp.distributions.MultivariateNormalFullCovariance(mu_neg, cov_neg)
    p = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(probs=[0.5, 0.5]),
        components=[p_pos, p_neg])

    q_pos = tfp.distributions.MultivariateNormalTriL(loc=mu_pos_rot, scale_tril=tf.linalg.cholesky(cov_pos_rot))
    q_neg = tfp.distributions.MultivariateNormalTriL(loc=mu_neg_rot, scale_tril=tf.linalg.cholesky(cov_neg_rot))
    #q_pos = tfp.distributions.MultivariateNormalFullCovariance(mu_pos_rot, cov_pos_rot)
    #q_neg = tfp.distributions.MultivariateNormalFullCovariance(mu_neg_rot, cov_neg_rot)
    q = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(probs=[0.5, 0.5]),
        components=[q_pos, q_neg])

    w_mog = IW_MoG(p, q)


    
    
        
    
