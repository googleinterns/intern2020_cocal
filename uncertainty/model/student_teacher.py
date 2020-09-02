import os, sys
import numpy as np

import tensorflow as tf

import model
from model.util import *


    
class PseudoLabeling:
    """Pseudo-labeling function, which is implemented in the python iterator"""

    def __init__(self, model_base, model_conf, ld_src, ld_tar, ideal=False):
        """
        model_base: a classifier
        model_conf: a confidence predictor
        ld_src: source data loader
        ld_tar: target data loader
        """
        self.model_base = model_base
        self.model_conf = model_conf
        self.ld_src = ld_src
        self.ld_tar = ld_tar
        self.ideal = ideal
        

    def __iter__(self):
        """Initialize iterators."""
        self.iter_end = {'src': False, 'tar': False}
        self.iter = {'src': iter(self.ld_src), 'tar': iter(self.ld_tar)}
        return self

    
    def __next__(self):
        """return the next labeled examples"""

        ## source
        def sample_from_source():
            return next(self.iter['src'])

        try:
            x, y = sample_from_source()
        except StopIteration:
            self.iter_end['src'] = True

            if self.iter_end['src'] and self.iter_end['tar']:
                raise StopIteration
            else:
                self.iter['src'] = iter(self.ld_src)
                x, y = sample_from_source()
        x_src, y_src = x, y

        ## target
        def sample_from_target():
            while True:
                x, y = next(self.iter['tar'])
                if len(x.shape) == 5:
                    x_tc, x_st = x[:, 0, :, :, :], x[:, 1, :, :, :] # two different augmentation
                else:
                    x_tc, x_st = x, x

                y_pred = self.model_base(x_tc)['y_pred']
                conf = self.model_conf(x_tc)
                
                if self.ideal:
                    x_conf_i, y_conf_i, y_true_i = x_st[conf==1], y[conf==1], y[conf==1]
                else:
                    x_conf_i, y_conf_i, y_true_i = x_st[conf==1], y_pred[conf==1], y[conf==1]

                if any(conf==1):
                    break


            ## upsample
            n_repeat = x_st.shape[0] // x_conf_i.shape[0]
            x_conf = tf.repeat(x_conf_i, n_repeat, 0)
            y_conf = tf.repeat(y_conf_i, n_repeat, 0)
            y_true = tf.repeat(y_true_i, n_repeat, 0)
            
            n_remain = x_st.shape[0] - x_conf.shape[0]
            if n_remain > 0:
                x_conf = tf.concat((x_conf, x_conf_i[:n_remain]), 0)
                y_conf = tf.concat((y_conf, y_conf_i[:n_remain]), 0)
                y_true = tf.concat((y_true, y_true_i[:n_remain]), 0)
                

            return x_conf, y_conf

        try:
            x, y = sample_from_target()
        except StopIteration:
            self.iter_end['tar'] = True

            if self.iter_end['src'] and self.iter_end['tar']:
                raise StopIteration
            else:
                self.iter['tar'] = iter(self.ld_tar)

                try:
                    x, y = sample_from_target()
                except StopIteration:
                    # it is possible that there are no confident examples
                    x, y = None, None
                    
        x_tar, y_tar = x, y
        
        
        ## merge
        if x_tar is not None:
            x = tf.concat((x_src, x_tar), 0)
            y = tf.concat((tf.cast(y_src, tf.int64), tf.cast(y_tar, tf.int64)), 0)
        else:
            x, y = x_src, y_src
        return x, y

        
class TargetPseudoLabeling:
    """Pseudo-labeling function, which is used only for target."""
    def __init__(self, model_base, model_conf, ld_tar, ideal=False):
        """
        model_base: a classifier
        model_conf: a confidence predictor
        ld_src: source data loader
        ld_tar: target data loader
        """
        
        self.model_base = model_base
        self.model_conf = model_conf
        self.ld_tar = ld_tar
        self.ideal = ideal
        

    def __iter__(self):
        """Initialize an iterator."""
        
        self.iter = iter(self.ld_tar)
        return self

    
    def __next__(self):
        """return the next labeled examples"""

        def sample_from_target():
            while True:
                x, y = next(self.iter)
                if len(x.shape) == 5:
                    x_tc, x_st = x[:, 0, :, :, :], x[:, 1, :, :, :]
                else:
                    x_tc, x_st = x, x
                y_pred = self.model_base(x_tc)['y_pred']
                conf = self.model_conf(x_tc) ##TODO: return dict?
                
                if self.ideal:
                    x_conf_i, y_conf_i, y_true_i = x_st[conf==1], y[conf==1], y[conf==1]
                else:
                    x_conf_i, y_conf_i, y_true_i = x_st[conf==1], y_pred[conf==1], y[conf==1]

                if any(conf==1):
                    break
                        
            return x_conf_i, y_conf_i

        return sample_from_target()



class Teacher(tf.keras.Model):
    """Teacher model, which includes a pseudo-labeling function. It acts like data loaders."""
    def __init__(self, params, model_base, ds_src, ds_tar, ideal=False):
        """
        Initialize a teacher model

        params: required model parmeters
        model_base: a classifier in the keras model
        ds_src: source dataset loader, which includes train/val/test loaders
        ds_tar: target dataset loader, which includes train/val/test loaders
        idea: If True, the pseudo-labeling function returns the true labels.
        """
        super().__init__()

        self.model_base = model_base
        self.ideal = ideal
        self.model_conf = getattr(model, params.conf)(self.model_base)
        
        self.train = PseudoLabeling(self.model_base, self.model_conf, ds_src.train, ds_tar.train, ideal)
        self.val = TargetPseudoLabeling(self.model_base, self.model_conf, ds_tar.val, ideal) ## target only
        self.test = TargetPseudoLabeling(self.model_base, self.model_conf, ds_tar.test, ideal) ## target only
        

class Student(Teacher):
    """Stduent model"""
    def __init__(self, params, model_base, ds_src, ds_tar, ideal=False):
        super().__init__(params, model_base, ds_src, ds_tar, ideal)
