import os, sys
from functools import partial
import tensorflow as tf

from learning import BaseLearner
from learning import precision

from data import plot_data

class LearnerConfPred(BaseLearner):
    """Confidence predictor learning class. The legacy code fill be REMOVED after testing."""
    def __init__(self, params, model, model_pred, model_cond_iw=None, model_name_postfix='_confpred'):
        super().__init__(params, model, model_name_postfix=model_name_postfix)
        self.model_pred = model_pred
        self.model_cond_iw = model_cond_iw

        self.loss_fn_train = precision
        self.loss_fn_val = precision
        self.loss_fn_test = precision


    def train(self, ld_tr, ld_val, ld_te=None):
        """This funciton will be REMOVED after testing."""
        ## load
        if not self.params.find_best and self._check_final():
            if self.params.load_final:
                self._load_final()
            else:
                self._load_best()
            return

        ## line search
        T_opt, prec_opt = 1.0, 1.0
        for T in tf.range(self.params.T_max, self.params.T_min, -self.params.T_step):
            
            self.model.T = tf.Variable(T)
            self.model_cond_iw.update_p_q_conf() ## update when update T
            
            # compute precision
            prec, n_conf = self.test(ld_tr, iw_fn=lambda x: self.model_cond_iw(x, training=False))
            
            msg = 'eps = %f, T = %f, prec = %f, n_conf = %d'%(self.params.eps, self.model.T, prec, n_conf)
            
            # check a condition
            if prec >= 1.0 - self.params.eps:
                T_opt, prec_opt = T, prec

                ## save the best
                self.model.T = tf.Variable(T_opt)
                self.model_cond_iw.update_p_q_conf() ## update when update T

                model_fn = os.path.join(self.params.save_root, 'model_params%s_best'%(self.model_name_postfix))
                self.model.save_weights(model_fn)
                msg += ', saved'
                
            print(msg)

        ## save the final
        model_fn = os.path.join(self.params.save_root, 'model_params%s_final'%(self.model_name_postfix))
        self.model.save_weights(model_fn)

        ## load the best
        self._load_best()


        
    def test(self, ld_te, loss_fn=None, iw_fn=None, ld_name='', verbose=False):
        """This function compute precision and coverage of the psuedo-labeling function."""
        # compute precision
        prec_vec = []
        n_conf = 0
        n = 0
        for x, y in ld_te:
            prec_i, n_conf_i = self.loss_fn_test(
                x, y,
                lambda x: self.model_pred(x, training=False)['logits'],
                self.model,
                model_iw=iw_fn, reduce='none')
            prec_vec.append(prec_i)
            n_conf += n_conf_i
            n += y.shape[0]
        prec = tf.math.reduce_mean(tf.cast(tf.concat(prec_vec, 0), tf.float32))
        if verbose:
            ## print
            print('[test%s] T = %f, precision = %.2f%%, size = %d/%d = %.2f%%'%(
                ld_name if ld_name is '' else ' on %s'%(ld_name),
                self.model.T if hasattr(self.model, 'T') else -1, prec*100.0,
                n_conf, n, float(n_conf)/float(n)*100.0))

            ## visualize for 2d data
            x_list = []
            y_list = []
            show = True
            for x, y in ld_te:
                if x.shape[-1] is not 2 or any(y>1):
                    show = False
                    break
                conf = self.model(x)
                i_conf = conf==1

                x_list.append(x)
                y_list.append(y+2*tf.cast(i_conf, tf.int64))

            if show:
                x_list = tf.concat(x_list, 0).numpy()
                y_list = tf.concat(y_list, 0).numpy()
                plot_data(
                    x_list,
                    y_list,
                    markers=['s', 's', 's', 's'],
                    colors=['r', 'g', 'k', 'k'],
                    facecolors=['r', 'g', 'r', 'g'],
                    alphas=[0.5, 0.5, 1.0, 1.0],
                    labels=[r'$-$', r'$+$', r'$-$'+' (conf)', r'$+$'+' (conf)'],
                    markersize=4,
                    linewidth=2,
                    classifier=lambda x: tf.nn.softmax(self.model_pred(tf.constant(x, dtype=tf.float32), training=False)['logits'], -1).numpy(),
                    fn=os.path.join(self.params.save_root, "conf_examples_%s"%(ld_name)),
                )
                
                
        return prec, n_conf, n
            
            
