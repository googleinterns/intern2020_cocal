import os, sys, shutil
import numpy as np

import tensorflow as tf

from data.util import *
from data.augment import compose_augment_seq

class MNIST(DataLoader):
    def __init__(self, root=None, batch_size=100,
                 train_shuffle=True, val_shuffle=False, test_shuffle=False,
                 train_aug=True, val_aug=False, test_aug=False,
                 size=32, color=True, val_ratio=0.5, aug_list=None, domain_id=None, sample_ratio=1.0, resize_pad=False, double_aug=False):

        assert(size>=28)
        
        ## default tforms
        tforms_dft = [
            Mat2RGB() if color else Mat2Gray(),
            #Pad([size-28, size-28]) if size>28 else Identity(),
            #Resize([size, size]) if size!=28 else Identity(),
            Identity() if size==28 else Pad([size-28, size-28]) if resize_pad else Resize([size, size]),
            Cast(tf.float32, tf.int32),
            Scaling(1.0/255.0),
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

        
        ##TODO: tfds?
        ## load data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        (x_val, y_val), (x_test, y_test) = self._split_data(x_test, y_test, val_ratio, True)

        ## reduce train/val samples (shuffle before sampling)
        x_train, y_train = shuffle_labeled_examples(x_train, y_train)
        x_train, y_train = x_train[:round(x_train.shape[0]*sample_ratio)], y_train[:round(y_train.shape[0]*sample_ratio)]
        x_val, y_val = shuffle_labeled_examples(x_val, y_val)
        x_val, y_val = x_val[:round(x_val.shape[0]*sample_ratio)], y_val[:round(y_val.shape[0]*sample_ratio)]

        ## init meta data
        self.n_train = y_train.shape[0]
        self.n_val = y_val.shape[0]
        self.n_test = y_test.shape[0]

        ## construct data loaders
        self.train = self._init_loader(tf.data.Dataset.from_tensor_slices((x_train, y_train)), self.n_train, train_shuffle, batch_size, tforms_train)
        self.val = self._init_loader(tf.data.Dataset.from_tensor_slices((x_val, y_val)), self.n_val, val_shuffle, batch_size, tforms_val)
        self.test = self._init_loader(tf.data.Dataset.from_tensor_slices((x_test, y_test)), self.n_test, test_shuffle, batch_size, tforms_test)
        if domain_id is not None:
            self.train_dom = self._init_loader(tf.data.Dataset.from_tensor_slices((x_train, y_train)), self.n_train, train_shuffle, batch_size, tforms_train_dom)
            self.val_dom = self._init_loader(tf.data.Dataset.from_tensor_slices((x_val, y_val)), self.n_val, val_shuffle, batch_size, tforms_val_dom)
            self.test_dom = self._init_loader(tf.data.Dataset.from_tensor_slices((x_test, y_test)), self.n_test, test_shuffle, batch_size, tforms_test_dom)
        

if __name__ == '__main__':
    aug_list = [
        ('jitter', {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4}),
        ('randaug', {'size': 32, 'mode': 'SHAKE'})
    ]
    dsld = MNIST(size=32, color=True, aug_list=aug_list)

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
    tmp_root = 'tmp/mnist/train'
    shutil.rmtree(tmp_root, ignore_errors=True)
    os.makedirs(tmp_root, exist_ok=True)
    for x, y in dsld.train:
        [tf.io.write_file(
            os.path.join(tmp_root, "%d_%d.png"%(i, y_b)),
            tf.image.encode_png(tf.cast(x_b*255, tf.uint8))) for i, (x_b, y_b) in enumerate(zip(x, y))]
        break
        

    
