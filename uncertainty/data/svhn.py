import os, sys, shutil
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from data.util import *
from data.augment import compose_augment_seq

class SVHN(DataLoader):
    def __init__(self, root=None, batch_size=100,
                 train_shuffle=True, val_shuffle=False, test_shuffle=False,
                 train_aug=True, val_aug=False, test_aug=False,
                 size=32, color=True, val_ratio=0.5, aug_list=None, domain_id=None, sample_ratio=1.0, resize_pad=False, double_aug=False):

        assert(sample_ratio == 1.0)
        assert(resize_pad == False)
        
        ## default tforms
        tforms_dft = [
            Map2Tuple(),
            Resize([size, size]) if size!=32 else Identity(),
            Identity() if color else RGB2Gray(),
            Float(),
            Normalize(0.5, 0.5),
            #PerImageStd(),
        ]

        ## data augmentation ops
        if aug_list is not None:
            tforms_aug = [DuplicateX() if double_aug else Identity()]
            tforms_aug += [aug_op for aug_op in compose_augment_seq(aug_list, is_training=True)]
        else:
            tforms_aug = []
            
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
        tf.random.set_seed(0)
        ds_train = tfds.load('svhn_cropped', split='train', shuffle_files=True)
        ds_val = tfds.load('svhn_cropped', split='test[:%d'%(round(val_ratio*100.))+'%]', shuffle_files=True)
        ds_test = tfds.load('svhn_cropped', split='test[%d'%(round(val_ratio*100.))+'%:]', shuffle_files=True)

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
    dsld = SVHN()

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
    tmp_root = 'tmp/svhn/train'
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)
    for x, y in dsld.train:
        [tf.io.write_file(
            os.path.join(tmp_root, "%d_%d.png"%(i, y_b)),
            tf.image.encode_png(tf.cast(x_b*255, tf.uint8))) for i, (x_b, y_b) in enumerate(zip(x, y))]
        break
        

    
