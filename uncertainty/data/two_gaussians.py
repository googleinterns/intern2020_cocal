import os, sys
import numpy as np
from data.util import *

import tensorflow as tf
import tensorflow_probability as tfp

class TwoGaussians(DataLoader):
    def __init__(self, batch_size,
                 n_pos, n_neg, mu_pos, cov_pos, mu_neg, cov_neg,
                 rot=0.0, train_ratio=0.8, val_ratio=0.1,
                 train_shuffle=True, val_shuffle=True, test_shuffle=False,
                 seed=0
    ):
        dim = len(mu_pos)
        assert(dim == cov_pos.shape[0] == len(mu_neg) == cov_neg.shape[0])

        rot_rad = np.deg2rad(rot)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]], dtype=np.float32)
        
        ## generate data
        np.random.seed(seed)
        x_pos = np.random.multivariate_normal(mu_pos, cov_pos, n_pos).astype(np.float32)
        x_neg = np.random.multivariate_normal(mu_neg, cov_neg, n_neg).astype(np.float32)
        y_pos = np.ones((n_pos), dtype=np.int64)
        y_neg = np.zeros((n_neg), dtype=np.int64)
        x = np.concatenate((x_pos, x_neg), 0)
        y = np.concatenate((y_pos, y_neg))
        x = np.transpose(np.matmul(R, np.transpose(x)))
        i_rnd = np.random.permutation(len(y))
        x = x[i_rnd]
        y = y[i_rnd]

        ## split
        n = len(y)
        n_train = int(n*train_ratio)
        n_val = int(n*val_ratio)
        n_test = n - n_train - n_val
        x_train, y_train = x[:n_train], y[:n_train]
        x_val, y_val = x[n_train:n_train+n_val], y[n_train:n_train+n_val]
        x_test, y_test = x[n_train+n_val:], y[n_train+n_val:]
        self.n_train, self.n_val, self.n_test = n_train, n_val, n_test

        ## init loaders
        self.train = self._init_loader(tf.data.Dataset.from_tensor_slices((x_train, y_train)), self.n_train, train_shuffle, batch_size)
        self.val = self._init_loader(tf.data.Dataset.from_tensor_slices((x_val, y_val)), self.n_val, val_shuffle, batch_size)
        self.test = self._init_loader(tf.data.Dataset.from_tensor_slices((x_test, y_test)), self.n_test, test_shuffle, batch_size)


if __name__ == '__main__':
    rot_deg = 60
    
    mu_pos = [+0.6, +0.6]
    mu_neg = [-0.6, -0.6]
    
    cov_pos = np.diag([0.01, 0.1])
    cov_neg = np.diag([0.01, 0.1])
    
    mu_pos_rot, cov_pos_rot = rot_gaussian(rot_deg, mu_pos, cov_pos)
    mu_neg_rot, cov_neg_rot = rot_gaussian(rot_deg, mu_neg, cov_neg)
    
    dsld_src = TwoGaussians(
        100, 
        n_pos=10000, n_neg=10000,
        mu_pos=mu_pos, cov_pos=cov_pos,
        mu_neg=mu_neg, cov_neg=cov_neg
    )

    dsld_tar = TwoGaussians(
        100, 
        n_pos=10000, n_neg=10000,
        mu_pos=mu_pos_rot, cov_pos=cov_pos_rot,
        mu_neg=mu_neg_rot, cov_neg=cov_neg_rot,
        seed=1,
    )
        
    x_tr_src, y_tr_src = [], []
    for x, y in dsld_src.train:
        x_tr_src.append(x.numpy())
        y_tr_src.append(y.numpy())
    x_tr_src = np.concatenate(x_tr_src, 0)
    y_tr_src = np.concatenate(y_tr_src)

    x_tr_tar, y_tr_tar = [], []
    for x, y in dsld_tar.train:
        x_tr_tar.append(x.numpy())
        y_tr_tar.append(y.numpy())
    x_tr_tar = np.concatenate(x_tr_tar, 0)
    y_tr_tar = np.concatenate(y_tr_tar)
    y_tr_tar[y_tr_tar==0] = 2
    y_tr_tar[y_tr_tar==1] = 3

    ## IW
    from model import IW_MoG
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


    ## plot data
    plot_data(
        np.concatenate((x_tr_src, x_tr_tar), 0),
        np.concatenate((y_tr_src, y_tr_tar)),
        ['s', 's', 's', 's'],
        ['orange', 'b', 'r', 'g'],
        [0.5, 0.5, 0.5, 0.5],
        [r'$-$'+' (src)', r'$+$'+' (src)', r'$-$'+' (tar)', r'$+$'+' (tar)'],
        fn="two_gaussian_tr",
        w=w_mog
    )
