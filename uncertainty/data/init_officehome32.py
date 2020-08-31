import os, sys
import numpy as np
import pickle
import zipfile
import cv2
import glob
import shutil

def split(root_src, root_tar, train_ratio, val_ratio):
    fns = np.array(glob.glob(os.path.join(root_src, '**', '*.jpg')))
    i_rnd = np.random.permutation(len(fns))
    fns = fns[i_rnd]
    
    n = len(fns)
    n_train = round(n*train_ratio)
    n_val = round(n*val_ratio)
    n_test = n - n_train - n_val

    fn_train = fns[:n_train]
    fn_val = fns[n_train:n_train+n_val]
    fn_test = fns[n_train+n_val:]

    labels = set([os.path.split(os.path.split(fn)[0])[1] for fn in fns])
    print(labels)
    print(len(labels))
    
    ## create label folders
    for name_split in ['train', 'val', 'test']:
        for l in labels:
            label_dir = os.path.join(root_tar, name_split, l)
            os.makedirs(label_dir, exist_ok=True)

    ## copy
    for name_split, fn_split in zip(['train', 'val', 'test'], [fn_train, fn_val, fn_test]):
        print(name_split, ":", len(fn_split))
        for fn_src in fn_split:
            fn_base = os.path.split(fn_src)[1]
            label = os.path.split(os.path.split(fn_src)[0])[1]
            fn_tar = os.path.join(root_tar, name_split, label, fn_base)
            os.makedirs(os.path.dirname(fn_tar), exist_ok=True)
            shutil.copyfile(fn_src, fn_tar)
            print(fn_tar)

                    
    


if __name__ == '__main__':
    root = '/home/sangdonp_google_com/Research/INTERN2020_COCAL/datasets/OfficeHomeDataset_10072016'
    names_dom = ['Art', 'Clipart', 'Product', 'RealWorld']
    seed = 0
    train_ratio = 0.6
    val_ratio = 0.2
    np.random.seed(seed)
    
    for ds_name in names_dom:
        subroot_src = os.path.join(root, ds_name)
        subroot_tar = ds_name
        split(subroot_src, subroot_tar, train_ratio, val_ratio)

        
        
    sys.exit()
        
    
    ## read labeled examples
    x_train, y_train = load_pickle(fn_train)
    x_val, y_val = load_pickle(fn_val)
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    
    ## shuffle
    i_tr_rnd = np.random.permutation(y_train.shape[0])
    i_val_rnd = np.random.permutation(y_val.shape[0])
    x_train, y_train = x_train[i_tr_rnd], y_train[i_tr_rnd]
    x_val, y_val = x_val[i_val_rnd], y_val[i_val_rnd]
    
    ## split test
    n_val = round(y_val.shape[0]*val_ratio)
    x_val, y_val, x_test, y_test = x_val[:n_val], y_val[:n_val], x_val[n_val:], y_val[n_val:]

    print("train: ", x_train.shape, y_train.shape)
    print("val: ", x_val.shape, y_val.shape)
    print("test: ", x_test.shape, y_test.shape)

    #cv2.imwrite('test.png', x_val[11])

    ## write to files
    for split_name, (x, y) in zip(['train', 'val', 'test'],
                                  [(x_train, y_train), (x_val, y_val), (x_test, y_test)]):
        dir_root = os.path.join(dir_target, split_name)
        print(split_name, ":", x.shape, y.shape)

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            dir_label = os.path.join(dir_root, "%d"%y_i)
            os.makedirs(dir_label, exist_ok=True)
            fn = os.path.join(dir_label, "%d.png"%i)
            cv2.imwrite(fn, x_i)
            print(fn)

        
