import os, sys, shutil
import numpy as np

import tensorflow as tf

from data.util import *
from data.augment import compose_augment_seq

class OfficeHome32(DataLoader):
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
            Resize((32, 32)),
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
        self.train = DataFolderLoader(os.path.join(root, 'train'), batch_size, train_shuffle, tforms_train, ext='jpg')
        self.val = DataFolderLoader(os.path.join(root, 'val'), batch_size, val_shuffle, tforms_val, ext='jpg')
        self.test = DataFolderLoader(os.path.join(root, 'test'), batch_size, test_shuffle, tforms_test, ext='jpg')
        if domain_id is not None:
            self.train_dom = DataFolderLoader(os.path.join(root, 'train'), batch_size, train_shuffle, tforms_train_dom, ext='jpg')
            self.val_dom = DataFolderLoader(os.path.join(root, 'val'), batch_size, val_shuffle, tforms_val_dom, ext='jpg')
            self.test_dom = DataFolderLoader(os.path.join(root, 'test'), batch_size, test_shuffle, tforms_test_dom, ext='jpg')


class Art32(OfficeHome32):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
class Clipart32(OfficeHome32):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
class Product32(OfficeHome32):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
class RealWorld32(OfficeHome32):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    
            
if __name__ == '__main__':

    def count_n(ds):
        
        n_train = 0.0
        for x, y in ds.train:
            n_train += y.shape[0]
        n_val = 0.0
        for x, y in ds.val:
            n_val += y.shape[0]
        n_test = 0.0
        for x, y in ds.test:
            n_test += y.shape[0]

        print("n = %d + %d + %d = %d"%(n_train, n_val, n_test, n_train+n_val+n_test))

    print('Art')
    ds = Art32('data/Art')
    count_n(ds)

    print('Clipart')
    ds = Clipart32('data/Clipart')
    count_n(ds)

    print('Product')
    ds = Product32('data/Product')
    count_n(ds)

    print('RealWorld')
    ds = RealWorld32('data/RealWorld')
    count_n(ds)

        
