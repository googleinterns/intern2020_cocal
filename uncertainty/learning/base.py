""" Basic neural network training class."""

import os, sys
import time
import numpy as np

import tensorflow as tf
from learning.loss import reg_l2


class BaseLearner:
    def __init__(self, params, model, model_iw=None, model_name_postfix=''):
        """Initialize the BaseLeaner class. 

        params: required parameters for training (e.g., learning rate)
        model: keras model
        model_iw: keras model for importance weight
        model_name_postfix: a model name postfix, which is used in saved model files
        """

        self.params = params
        self.model = model
        self.model_iw = model_iw
        self.loss_fn_train = None
        self.loss_fn_val = None
        self.loss_fn_test = None
        self.model_name_postfix = model_name_postfix

        if hasattr(self.params, 'save_root'):
            self.model_fn_best = os.path.join(self.params.save_root, 'model_params%s_best'%(self.model_name_postfix))
            self.model_fn_final = os.path.join(self.params.save_root, 'model_params%s_final'%(self.model_name_postfix))

        
    def _load_best(self):
        """Load the best keras model."""
        self.model.load_weights(self.model_fn_best)
        print("[load] the best model is loaded")

        
    def _check_best(self):
        """Check whether there exists files for the best keras model."""
        return os.path.exists(self.model_fn_best + '.index')

        
    def _load_final(self):
        """Load the final keras model."""
        self.model.load_weights(self.model_fn_final)
        print("[load] the final model is loaded")


    def _check_final(self):
        """Check whether there exists files for the final keras model."""
        return os.path.exists(self.model_fn_final + '.index')

            
    def _train_begin(self, ld_val):
        """A function called in the begning of training process. It initializes an optimizer, and save an initial model as the initial best model.

        ld_val: validation dataset loader
        """

        ## init an optimizer
        self.lr = self.params.lr
        if self.params.optim == 'SGD':
            self.opt = tf.keras.optimizers.SGD(learning_rate=lambda: self.lr, momentum=self.params.momentum)
        elif self.params.optim == 'Adam':
            self.opt = tf.keras.optimizers.Adam(learning_rate=lambda: self.lr)
        else:
            raise NotImplementedError

        ## save the initial model
        if ld_val is not None:
            self.error_val, *_ = self.validate(ld_val)
            model_fn = os.path.join(self.params.save_root, 'model_params%s_best'%(self.model_name_postfix))
            self.model.save_weights(model_fn)

    
    def _train_end(self):
        """A function called at the end of training process. It saves the final model and load final/best models, depending on an option."""
        ## save the final model
        model_fn = os.path.join(self.params.save_root, 'model_params%s_final'%(self.model_name_postfix))
        self.model.save_weights(model_fn)
        print("[final model] saved at %s"%(model_fn))

        ## load the best model
        if self.params.load_final:
            self._load_final()
        else:
            self._load_best()

        
    def _train_epoch_begin(self, i_epoch):
        """A function called at the begining of each epoch. It measures epoch duration, and update learning rate."""
        self.t_epoch_begin = time.time()
        
        ## update learning rate
        self.lr = self.params.lr*self.params.lr_step_decay_rate**((i_epoch-1)//self.params.lr_step_size)
        

    def _train_epoch_end(self, i_epoch, ld_val, ld_te=None):
        """A function called at the end of each epoch. It prints logs and valiate the current model.
        
        i_epoch: current epoch id
        ld_val: validation dataset loader
        ld_te: test dataset loader
        """

        ## print progress
        msg = "[epoch: %d/%d, %.2f sec"%(i_epoch, self.params.n_epochs, time.time() - self.t_epoch_begin)
        ## print lr
        if hasattr(self, 'lr'):
            msg += ", lr = %.2e"%(self.lr)
        ## print reg params
        for k, v in self.__dict__.items():
            if 'reg_' in k:
                msg += ', %s = %.2e'%(k, v)
        msg += '] '
        
        ## print losses
        for k, v, in self.__dict__.items():
            if 'loss' in k and '_fn' not in k:
                msg += '%s = %.4f, '%(k, v)
        
        ## validate the current model
        if i_epoch % self.params.val_period == 0:
            if ld_te is not None:
                ##TODO: assume classification
                error_te, ece_te, ece_oc_te, *_ = self.test(ld_te)
                msg += 'error_test = %.2f%%, ECE_test = %.2f%% (%.2f%%), '%(error_te*100.0, ece_te*100.0, ece_oc_te*100.0)
            if ld_val is not None:
                error_val, *_ = self.validate(ld_val)
                msg += 'error_val = %.4f (error_val_best = %.4f)'%(error_val, self.error_val)
            
        if ld_val is not None:
            if self.error_val >= error_val:
                ## save the best model
                self.error_val = error_val
                model_fn = os.path.join(self.params.save_root, 'model_params%s_best'%(self.model_name_postfix))
                self.model.save_weights(model_fn)
                msg += ', saved!'
        else:
            ## save the final model
            model_fn = os.path.join(self.params.save_root, 'model_params%s_final'%(self.model_name_postfix))
            self.model.save_weights(model_fn)
            msg += ', saved!'

        print(msg)

        
    def _train_epoch(self, ld_tr):
        """A function called at each epoch. It conducts optimization.

        ld_tr: training data loader

        """

        ld_tr = [ld_tr] if not isinstance(ld_tr, list) or len(ld_tr)==0 else ld_tr
        for data in zip(*ld_tr):
            x = [d[0] for d in data]
            x = x[0] if len(x)==1 else x
            y = [d[1] for d in data]
            y = y[0] if len(y)==1 else y

            #st = time.time()
            with tf.GradientTape() as tape:
                loss_dict = self.loss_fn_train(x, y,
                                               model=lambda x: self.model(x, training=True),
                                               model_iw=self.model_iw, tape=tape)
                [setattr(self, k, v) for k, v in loss_dict.items()]
                
                if self.params.weight_decay > 0.0:
                    for v in self.model.trainable_weights:
                        if 'bn' in v.name:
                            print(v)
                            print('not to decay')
                            sys.exit()
                    self.loss_wd = self.params.weight_decay * reg_l2([v for v in self.model.trainable_weights if 'kernel' in v.name])
                    self.loss += self.loss_wd

            #print(time.time() - st)
            grad = tape.gradient(self.loss, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grad, self.model.trainable_weights))

        
    def train(self, ld_tr, ld_val, ld_te=None):
        """The main training function.

        ld_tr: training dataset loader
        ld_val: validation dataset loader
        ld_te: test dataset loader
        """
        if not self.params.rerun and self._check_final():
            if self.params.load_final:
                self._load_final()
            else:
                self._load_best()
            return

        self._train_begin(ld_val)
        for i in range(1, self.params.n_epochs+1):
            self._train_epoch_begin(i)
            self._train_epoch(ld_tr)
            self._train_epoch_end(i, ld_val, ld_te)
        self._train_end()
            
        
    def validate(self, ld_val, verbose=False):
        """This funciton validate the current model using validation dataset loader.

        ld_val: validation dataset loader
        verbose: False if you don't need to see logs
        """
        return self.test(ld_val, loss_fn=self.loss_fn_val, iw_fn=self.model_iw, verbose=verbose)

    
    def test(self, ld_te, loss_fn=None, iw_fn=None, verbose=False):
        """This function test the current model using the given dataloader

        ld_te: test dataset loader in general, but can be anything
        loss_fn: adesired loss function to test
        iw_fn: importance weigh tfunction
        verbose: True if you want to see logs
        """
        loss_fn = self.loss_fn_test if loss_fn is None else loss_fn
        loss_vec = []
        #ph = []
        
        for x, y in ld_te:
            loss_dict = loss_fn(x, y,
                                model=lambda x: self.model(x, training=False),
                                model_iw=iw_fn, reduce='none')
            loss_vec.append(loss_dict['loss'])

        if loss_vec:
            loss_vec = tf.concat(loss_vec, 0)
            loss = tf.math.reduce_mean(loss_vec)
        else:
            print("!! base.test(): no examples from the loader, thus loss = np.inf")
            loss = np.inf

        return loss, loss_vec

            
        
