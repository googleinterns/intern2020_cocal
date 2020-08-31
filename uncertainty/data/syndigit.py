import os, sys, shutil
import numpy as np
import glob

import tensorflow as tf

from data.util import *
from data.augment import compose_augment_seq

class SynDigit(DataLoader):
    def __init__(self, root, batch_size=100,
                 train_shuffle=True, val_shuffle=False, test_shuffle=False,
                 train_aug=True, val_aug=False, test_aug=False,
                 size=32, val_ratio=0.5, aug_list=None, domain_id=None):

        ## default tforms
        tforms_dft = [
            File2Tensor(3), 
            Resize([size, size]),
            Cast(tf.float32, tf.int32),
            Scaling(1.0/255.0),
            Normalize(0.5, 0.5),
        ]

        ## data augmentation ops
        tforms_aug = [aug_op for aug_op in compose_augment_seq(aug_list, is_training=True)] if aug_list is not None else []

        ## domain label
        tforms_dom = [DomainLabel(domain_id)] if domain_id is not None else []

        ## tforms
        tforms_train = tforms_dft + tforms_aug if train_aug else tforms_dft
        tforms_val = tforms_dft + tforms_aug if val_aug else tforms_dft
        tforms_test = tforms_dft + tforms_aug if test_aug else tforms_dft
        if domain_id is not None:
            tforms_train_dom = tforms_train + tforms_dom
            tforms_val_dom = tforms_val + tforms_dom
            tforms_test_dom = tforms_test + tforms_dom

        ## load data
        np.random.seed(0)
        tr_fns = glob.glob(os.path.join(root, 'imgs_train', '*/*.jpg'))
        te_fns = glob.glob(os.path.join(root, 'imgs_valid', '*/*.jpg'))
        np.random.shuffle(tr_fns)
        np.random.shuffle(te_fns)
        val_fns = te_fns[:round(len(te_fns)*val_ratio)]
        te_fns = te_fns[round(len(te_fns)*val_ratio):]
        
        ds_train = tf.data.Dataset.from_tensor_slices(tr_fns)
        ds_val = tf.data.Dataset.from_tensor_slices(val_fns)
        ds_test = tf.data.Dataset.from_tensor_slices(te_fns)
        
        ## construct data loaders
        self.train = self._init_loader(ds_train, batch_size*10, train_shuffle, batch_size, tforms_train)
        self.val = self._init_loader(ds_val, batch_size*10, val_shuffle, batch_size, tforms_val)
        self.test = self._init_loader(ds_test, batch_size*10, test_shuffle, batch_size, tforms_test)
        if domain_id is not None:
            self.train_dom = self._init_loader(ds_train, batch_size*10, train_shuffle, batch_size, tforms_train_dom)
            self.val_dom = self._init_loader(ds_val, batch_size*10, val_shuffle, batch_size, tforms_val_dom)
            self.test_dom = self._init_loader(ds_test, batch_size*10, test_shuffle, batch_size, tforms_test_dom)
            
        

if __name__ == '__main__':
    # aug_list = [
    #     ('jitter', {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4}),
    #     ('randaug', {'size': 32, 'mode': 'SHAKE'})
    # ]
    dsld = SynDigit('data/synthetic_digits')

    ## print #train
    n_train = sum([y.shape[0] for _, y in dsld.train])
    print("#train = %d"%(n_train))
    
    ## print #val
    n_val = sum([y.shape[0] for _, y in dsld.val])
    print("#val = %d"%(n_val))

    ## print #test
    n_test = sum([y.shape[0] for _, y in dsld.test])
    print("#test = %d"%(n_test))

    ## generate a few examples
    tmp_root = 'tmp/syndigit/train'
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)
    for x, y in dsld.train:
        [tf.io.write_file(
            os.path.join(tmp_root, "%d_%d.png"%(i, y_b)),
            tf.image.encode_png(tf.cast(x_b*255, tf.uint8))) for i, (x_b, y_b) in enumerate(zip(x, y))]
        break
        

    
