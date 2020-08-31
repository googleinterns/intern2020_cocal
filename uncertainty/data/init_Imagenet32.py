import os, sys
import numpy as np
import pickle
import zipfile
import cv2


def load_pickle(fn):
    examples, labels = [], []
    with zipfile.ZipFile(fn, 'r') as myzip:
        for fn_i in myzip.namelist():
            with myzip.open(fn_i) as f:
                dict = pickle.load(f)
            
                examples_i, labels_i = dict['data'], np.array(dict['labels'])
                examples_i = np.reshape(examples_i, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
                examples.append(examples_i)
                labels.append(labels_i)
    examples = np.concatenate(examples, 0)
    labels = np.concatenate(labels, 0)
    return examples, labels        



if __name__ == '__main__':
    fn_train = '/home/sangdonp_google_com/Research/INTERN2020_COCAL/datasets/Imagenet12_32x32/Imagenet32_train.zip'
    fn_val = '/home/sangdonp_google_com/Research/INTERN2020_COCAL/datasets/Imagenet12_32x32/Imagenet32_val.zip'
    dir_target = 'imagenet32'
    seed = 0
    val_ratio = 0.5
    
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

        
