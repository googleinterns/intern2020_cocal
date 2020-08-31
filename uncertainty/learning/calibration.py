import os, sys
from functools import partial
import tensorflow as tf

from learning import LearnerCls
from learning import loss_xe, loss_01

class TempScalingCls(LearnerCls):
    """Temperature scaling for classificaiton. The algorithm is the same as classification learning excecpt for using cross-entropy loss for valiadation."""
    def __init__(self, params, model, model_iw=None, model_name_postfix='_cal'):
        super().__init__(params, model, model_iw, model_name_postfix=model_name_postfix)
        self.loss_fn_train = loss_xe
        self.loss_fn_val = loss_xe
        self.loss_fn_test = loss_01

           
            

        

