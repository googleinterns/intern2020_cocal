import os, sys
import time
import numpy as np

import tensorflow as tf
import model

from learning import LearnerCls, LearnerDACls, LearnerConfPred
from learning import TempScalingCls as CalibratorCls

class BaseLearnerSelf:
    def __init__(self, params, params_base, model_s, model_t, params_conf=None, params_init=None, model_name_postfix=''):
        """Initialize the basic self-training class. 

        params: required parameters for self-training (e.g., learning rate)
        params_base: parameters for base model training 
        model_s: keras student model
        model_t: keras teacher model
        params_conf: parameters for a confidence classifier
        params_init: parameters for initial model training
        model_name_postfix: a model name postfix, which is used in saved model files
        """

        self.params = params
        self.params_base = params_base
        self.params_init = params_init
        self.params_conf = params_conf
        self.model_s = model_s
        self.model_t = model_t
        self.model = self.model_s.model_base
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
        

    def _train_begin(self, ds_src, ds_tar, ds_dom):
        """A function called in the begning of training process. It trains an initial model for self-training.

        ds_src: source dataset loader, which contains loaders for train/val/test
        ds_tar: target dataset loader, which contains loaders for train/val/test
        ds_dom: domain dataset loader, which contains loaders for train/val/test
        """

        ## initialize a base model
        if self.params.init_advtr:
            print("## init the teacher model with adversarial training")
            model_base = self.model
            model.set_trainable(model_base, True)

            ## init a adv model
            mdl_adv = getattr(model, self.params_init.model_advtr)(n_in=model_base.dim_feat)
            ## init a learner
            learner = LearnerDACls(self.params_init, model.DAN(model_base, mdl_adv), model_name_postfix='_advtrinit')
            ## train the model
            learner.train([ds_src.train, ds_dom.train], None, ds_tar.test)        
            ## test the model
            learner.test(ds_tar.test, ld_name='tar', verbose=True)
            print()            

        else:

            print("## init the teacher model with sourceonly training")
            model_base = self.model
            model.set_trainable(model_base, True)

            ## init a learner
            learner = LearnerCls(self.params_init, model_base, model_name_postfix='_sourceonlyinit')
            ## train the model
            learner.train(ds_src.train, ds_src.val, ds_tar.test)
            ## test the model
            learner.test(ds_tar.test, ld_name='tar', verbose=True)
            print()

            
        print('## initialize a pseudo-labeling function')
        model_base = self.model
        model_conf = self.model_s.train.model_conf
        ## init a learner
        learner = LearnerConfPred(self.params_conf, model_conf, model_base, None, model_name_postfix='_confpredinit')
        ## set a constant
        model_conf.T = tf.Variable(1.0 - self.params_conf.eps) 
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        print()

                
    
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
        """A function called at the begining of each epoch. It measures epoch duration, switch student and teacher, and save the initial student.

        i_epoch: the current epoch 
        """
 
        self.t_epoch_begin = time.time()
        self.i_epoch = i_epoch

        ## switch student and teacher
        print("## switch: student <-> teacher")
        self.model_t, self.model_s = self.model_s, self.model_t
        self.model = self.model_s.model_base # initialize the student model with the teacher model

        # ## student copy instead of swithcing
        # tmp_mdl_fn = os.path.join(self.params.save_root, 'model_params_tmp')
        # self.model_t.model_base.save_weights(tmp_mdl_fn)
        # self.model.load_weights(tmp_mdl_fn)
        print()

        ## save the initial model
        if i_epoch == 1:
            ## save the initial model
            self.error_val, *_ = self.validate(self.model_t.val)
            model_fn = os.path.join(self.params.save_root, 'model_params%s_best'%(self.model_name_postfix))
            self.model.save_weights(model_fn)
            print("[initial model] saved at %s"%(model_fn))
            print()
            

        
    def _train_epoch_end(self, i_epoch, ld_val, ld_te=None):
        """A function called at the end of each epoch. It prints logs and valiate the current model.
        
        i_epoch: current epoch id
        ld_val: validation dataset loader
        ld_te: test dataset loader
        """

        ## print progress
        msg = "[epoch: %d/%d, %.2f sec]"%(i_epoch, self.params.n_epochs, time.time() - self.t_epoch_begin)
        
        ## print losses
        for k, v, in self.__dict__.items():
            if 'loss' in k and '_fn' not in k:
                msg += '%s = %.4f, '%(k, v)
        
        ## validate the current model
        if i_epoch % self.params.val_period == 0:
            if ld_te is not None:
                error_te, ece_te, ece_oc_te, *_ = self.test(ld_te)
                msg += 'error_test = %.4f, ECE_test = %.2f%% (%.2f%%), '%(error_te, ece_te*100.0, ece_oc_te*100.0)
            error_val, *_ = self.validate(ld_val)
            msg += 'error_val = %.4f (error_val_best = %.4f)'%(error_val, self.error_val)
            
        ## save the best model
        if self.error_val >= error_val:
            self.error_val = error_val
            model_fn = os.path.join(self.params.save_root, 'model_params%s_best'%(self.model_name_postfix))
            self.model.save_weights(model_fn)
            msg += ', saved!'

        print(msg)

        
    def _train_epoch(self, ds_src, ds_tar, ds_dom):
        """A function called at each epoch. It setup a pseudo-labeling function, and learn a student model using a teacher.

        ds_src: source dataset loader, which contains loaders for train/val/test
        ds_tar: target dataset loader, which contains loaders for train/val/test
        ds_dom: domain dataset loader, which contains loaders for train/val/test
        """

        ## pick a confidence threshold
        model_base = self.model_t.train.model_base
        model_conf = self.model_t.train.model_conf
            
        ## init a learner
        print('## init a pseudo-labeling function')
        learner = LearnerConfPred(self.params_conf, model_conf, model_base, None, model_name_postfix='_confpred_epoch_%d'%(self.i_epoch))
        ## set a constant
        model_conf.T = tf.Variable(1.0 - self.params_conf.eps) 
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        learner.test(ds_tar.train, ld_name='tar (train)', verbose=True)
        print()

        ## learn the student model using the teacher
        print('## learn the student model')
        model_s = self.model
        model.set_trainable(model_s, True)
        ## init a learner
        learner = LearnerCls(self.params_base, model_s, model_name_postfix='_base_epoch_%d'%(self.i_epoch))
        learner.loss_fn_train = self.loss_fn_train ## use the same training loss
        ## train the model
        learner.train(self.model_t.train, self.model_t.val, ds_tar.test)
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        print()

        
    def train(self, ds_src, ds_tar, ds_dom, ds_src_init, ds_tar_init, ds_dom_init):
        """The main training function.

        ds_src: source dataset loader, which contains loaders for train/val/test
        ds_tar: target dataset loader, which contains loaders for train/val/test
        ds_dom: domain dataset loader, which contains loaders for train/val/test
        ds_src_init: source dataset loader to train an initial model, which contains loaders for train/val/test
        ds_tar_init: target dataset loader to train an initial model, which contains loaders for train/val/test
        ds_dom_init: domain dataset loader to train an initial model, which contains loaders for train/val/test
        """

        if not self.params.rerun and self._check_final():
            if self.params.load_final:
                self._load_final()
            else:
                self._load_best()
            return

        self._train_begin(ds_src_init, ds_tar_init, ds_dom_init)
        for i in range(1, self.params.n_epochs+1):
            self._train_epoch_begin(i)
            self._train_epoch(ds_src, ds_tar, ds_dom)
            self._train_epoch_end(i, self.model_t.val, ds_tar.test)
        self._train_end()
            
        
    def validate(self, ld_val):
        """This funciton validate the current model using validation dataset loader.

        ld_val: validation dataset loader
        """
 
        return self.test(ld_val, loss_fn=self.loss_fn_val)

    
    def test(self, ld_te, loss_fn=None, ld_name='', verbose=False):
        """This function test the current model using the given dataloader
                
        ld_te: test dataset loader in general, but can be anything
        loss_fn: adesired loss function to test
        iw_fn: importance weigh tfunction
        verbose: True if you want to see logs
        """

        loss_fn = self.loss_fn_test if loss_fn is None else loss_fn
        loss_vec = []
        
        for x, y in ld_te:
            loss_dict = loss_fn(x, y,
                                model=lambda x: self.model(x, training=False),
                                reduce='none')
            loss_vec.append(loss_dict['loss'])
        loss_vec = tf.concat(loss_vec, 0)
        loss = tf.math.reduce_mean(loss_vec)
        return loss, loss_vec

            
        
