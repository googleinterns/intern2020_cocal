import os, sys, shutil
import numpy as np

import tensorflow as tf

from data.util import *
from data.augment import compose_augment_seq

class Imagenet32(DataLoader):
    def __init__(self, root, batch_size=100,
                 train_shuffle=True, val_shuffle=False, test_shuffle=False,
                 train_aug=True, val_aug=False, test_aug=False,
                 color=True, size=32,
                 aug_list=None, domain_id=None):
        assert(size==32)
        ## default tforms
        tforms_dft = [
            Cast(tf.float32, tf.int32),
            Scaling(1.0/255.0),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

        ## default augment
        tforms_size_aug_rnd = [
            #RandomCrop((28, 28, 3)),
            RandomHorizontalFlip(),
        ]
        tforms_size_aug_no_rnd = [
            #CenterCrop(28, 32),
        ]            

        ## data augmentation ops
        tforms_aug = [aug_op for aug_op in compose_augment_seq(aug_list, is_training=True)] if aug_list is not None else []

        ## domain label
        tforms_dom = [DomainLabel(domain_id)] if domain_id is not None else []

        ## tforms
        tforms_train = tforms_dft + tforms_aug if train_aug else tforms_dft
        tforms_train += tforms_size_aug_rnd if train_shuffle else tforms_size_aug_no_rnd
        tforms_val = tforms_dft + tforms_aug if val_aug else tforms_dft
        tforms_val += tforms_size_aug_rnd if val_shuffle else tforms_size_aug_no_rnd
        tforms_test = tforms_dft + tforms_aug if test_aug else tforms_dft
        tforms_test += tforms_size_aug_rnd if test_shuffle else tforms_size_aug_no_rnd
        if domain_id is not None:
            tforms_train_dom = tforms_train + tforms_dom
            tforms_val_dom = tforms_val + tforms_dom
            tforms_test_dom = tforms_test + tforms_dom

        ## construct data loaders
        self.train = DataFolderLoader(os.path.join(root, 'train'), batch_size, train_shuffle, tforms_train)
        self.val = DataFolderLoader(os.path.join(root, 'val'), batch_size, val_shuffle, tforms_val)
        self.test = DataFolderLoader(os.path.join(root, 'test'), batch_size, test_shuffle, tforms_test)
        if domain_id is not None:
            self.train_dom = DataFolderLoader(os.path.join(root, 'train'), batch_size, train_shuffle, tforms_train_dom)
            self.val_dom = DataFolderLoader(os.path.join(root, 'val'), batch_size, val_shuffle, tforms_val_dom)
            self.test_dom = DataFolderLoader(os.path.join(root, 'test'), batch_size, test_shuffle, tforms_test_dom)


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
        

    
