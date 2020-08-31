import os, sys
import time
import math

import numpy as np
import tensorflow as tf

from learning import BaseLearnerSelf
from learning import loss_xe, loss_01, loss_self
from uncertainty import ECE, plot_acc_rank, plot_acc_conf, bino_ci

class LearnerClsSelf(BaseLearnerSelf):
    """self-training for classification."""

    def __init__(self, params, params_base, model_s, model_t, params_conf=None, params_init=None, model_name_postfix=''):
        """Initialize the class of self-training for classification. 

        params: required parameters for self-training (e.g., learning rate)
        params_base: parameters for base model training 
        model_s: keras student model
        model_t: keras teacher model
        params_conf: parameters for a confidence classifier
        params_init: parameters for initial model training
        model_name_postfix: a model name postfix, which is used in saved model files
        """

        super().__init__(params, params_base, model_s, model_t, params_conf=params_conf, params_init=params_init, model_name_postfix=model_name_postfix)
        #self.loss_fn_train = loss_self
        self.loss_fn_train = loss_xe
        self.loss_fn_val = loss_01
        self.loss_fn_test = loss_01


    def test(self, ld_te, loss_fn=None, ld_name='', verbose=False):
        """This function tests the current model using the given dataloader. This function is overridden to print calibration error.

        ld_te: test dataset loader in general, but can be anything
        loss_fn: adesired loss function to test
        ld_name: loader name to print
        verbose: True if you want to see logs
        """

        error, error_vec = super().test(ld_te, loss_fn=loss_fn)
        nll, *_ = super().test(ld_te, loss_fn=loss_xe)

        ## compute confidence intervals
        if loss_fn == loss_01:
            m = tf.math.reduce_mean(error_vec).numpy()
            error_vec = tf.cast(error_vec, tf.int64)
            k = tf.math.reduce_sum(error_vec).numpy()
            n = error_vec.shape[0]
            l, u = bino_ci(k, n)
            l = 0.0 if math.isnan(l) else l
            u = 1.0 if math.isnan(u) else u
            
            print(f'[DBG, LearnerClsSelf] val error: k={k}, n={n}, lower={l}, mean={m}, upper={u}')
            error = u

              
        ## compute calibration error
        log_ph_list, ph_list, yh_list, y_list = [], [], [], []
        for x, y in ld_te:
            logits = self.model(x, training=False)['logits']

            log_ph = tf.math.reduce_max(tf.nn.log_softmax(tf.cast(logits, tf.float64), -1), -1) # use log/float64 to avoid tie in sorting due to numerical error
            ph = tf.math.reduce_max(tf.nn.softmax(logits, -1), -1)
            yh = tf.math.argmax(logits, -1)
            log_ph_list.append(log_ph.numpy())
            ph_list.append(ph.numpy())
            yh_list.append(yh.numpy())
            y_list.append(y.numpy())
        log_ph_list = np.concatenate(log_ph_list)
        ph_list = np.concatenate(ph_list)
        yh_list = np.concatenate(yh_list)
        y_list = np.concatenate(y_list)
        ece, ece_oc = ECE(ph_list, yh_list, y_list,
                          overconf=True,
                          rel_diag_fn=os.path.join(self.params.save_root, 'rel_diag_%s%s'%(ld_name, self.model_name_postfix)) if verbose and hasattr(self.params, 'save_root') else None)
        if verbose:
            print('[test%s] cls error = %.2f%%, NLL = %f, ECE = %.2f%% (%.2f%%)'%(
                ld_name if ld_name is '' else ' on %s'%(ld_name),
                error*100.0, nll,ece*100.0, ece_oc*100.0))

            if hasattr(self.params, 'save_root'):
                ## plot the accuracy-ranking curve
                plot_acc_rank((yh_list==y_list).astype(np.float32), log_ph_list, fn=os.path.join(self.params.save_root, 'acc_rank_zoom_%s%s'%(ld_name, self.model_name_postfix)))
                plot_acc_rank((yh_list==y_list).astype(np.float32), log_ph_list, fn=os.path.join(self.params.save_root, 'acc_rank_%s%s'%(ld_name, self.model_name_postfix)), ratio=1.0)

                ## plot theh acccuracy-confidence curve
                plot_acc_conf((y_list==yh_list).astype(np.float32), np.exp(log_ph_list), fn=os.path.join(self.params.save_root, 'acc_conf_%s%s'%(ld_name, self.model_name_postfix)))


        return error, ece, ece_oc
