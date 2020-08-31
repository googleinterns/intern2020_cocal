import os, sys
import numpy as np
import itertools
import glob
import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf

import data


    
'''
data transformations
'''
        
class Identity:
    def __call__(self, x, y):
        return x, y


class Mat2Gray:
    def __call__(self, x, y):
        x = tf.expand_dims(x, -1)
        return x, y

    
class Mat2RGB:
    def __call__(self, x, y):
        x = tf.expand_dims(x, -1)
        x = tf.repeat(x, 3, -1)
        return x, y

    
class RGB2Gray:
    def __call__(self, x, y):
        x = tf.image.rgb_to_grayscale(x)
        return x, y


class Float:
    def __call__(self, x, y):
        x = tf.cast(x, tf.float32)/255.
        return x, y

    
class Cast:
    def __init__(self, x_type, y_type):
        self.x_type = x_type
        self.y_type = y_type

    def __call__(self, x, y):
        if self.x_type is not None:
            x = tf.cast(x, self.x_type)
        if self.y_type is not None:
            y = tf.cast(y, self.y_type)
        return x, y

    
class Scaling:
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, x, y):
        x = x * self.scaling_factor
        return x, y
    
    
class Resize:
    def __init__(self, size):
        self.hw = size
        
    def __call__(self, x, y):
        # try:
        #     x = tf.image.resize_with_pad(x, self.hw[0], self.hw[1])
        # except tf.python.framework.errors_impl.InvalidArgumentError:
        x = tf.image.resize(x, self.hw) 
        return x, y


class Pad:
    def __init__(self, pad_size):
        assert(pad_size[0]%2==0 and pad_size[1]%2==0)
        self.pad_size = pad_size


    def __call__(self, x, y):
        padding = tf.constant([[self.pad_size[0]//2, self.pad_size[0]//2], [self.pad_size[1]//2, self.pad_size[1]//2], [0, 0]])
        x = tf.pad(x, padding)
        return x, y


class Map2Tuple:
    def __call__(self, m, x_name='image', y_name='label'):
        return m[x_name], m[y_name]

    
class File2Tensor:
    def __init__(self, n_channels):
        self.n_channels = n_channels
        
    def _get_image(self, fn):
        img = tf.io.read_file(fn)
        img = tf.image.decode_jpeg(img, channels=self.n_channels)
        return img

    def _get_label(self, fn):
        ##TODO: assume a numeric label in the file path
        label = tf.strings.to_number(tf.strings.split(fn, '/')[-2], tf.int32)
        return label
    
    def __call__(self, fn):
        x = self._get_image(fn)
        y = self._get_label(fn)
        return x, y
    

class PerImageStd:
    def __call__(self, x, y):
        x = tf.image.per_image_standardization(x)
        return x, y

    
class DomainLabel:
    def __init__(self, domain_id):
        self.domain_id = tf.constant(domain_id)
        
    def __call__(self, x, y):
        y = self.domain_id
        return x, y

    
class Squeeze:
    def __call__(self, x, y):
        x = tf.squeeze(x, axis=0)
        y = tf.squeeze(y, axis=0)
        return x, y

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, x, y):
        x = (x - self.mean) / self.std
        return x, y
        

class PNGFile2LabeledExample:
    def __init__(self, str2label, n_channels):
        self.str2label = str2label
        self.n_channels = n_channels

        
    def __call__(self, fn, y):
        x = tf.io.read_file(fn)
        x = tf.image.decode_png(x, channels=self.n_channels)
        return x, y

class JPGFile2LabeledExample:
    def __init__(self, str2label, n_channels):
        self.str2label = str2label
        self.n_channels = n_channels

        
    def __call__(self, fn, y):
        x = tf.io.read_file(fn)
        x = tf.image.decode_jpeg(x, channels=self.n_channels)
        return x, y

class RandomCrop:
    def __init__(self, size):
        self.size = size

        
    def __call__(self, x, y):
        x = tf.image.random_crop(x, self.size)
        return x, y


class RandomHorizontalFlip:
    def __call__(self, x, y):
        x = tf.image.random_flip_left_right(x)
        return x, y

class CenterCrop:
    def __init__(self, size, size_ori):
        self.size = size
        self.size_ori = size_ori

    def __call__(self, x, y):
        x = tf.image.central_crop(x, float(self.size)/float(self.size_ori))
        return x, y


class DuplicateX:
    def __call__(self, x, y):
        x1 = tf.identity(x)
        x2 = tf.identity(x)
        return [x1, x2], y

    
"""
loaders
"""
class DataLoader:

    def _init_loader(self, dataset, n, shuffle, batch_size, tforms=[]):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(n, reshuffle_each_iteration=True)
        for tform in tforms:
            dataset = dataset.map(tform, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    
    def _split_data(self, x, y, ratio, random=True, seed=0):
        assert(x.shape[0] == y.shape[0])
        idx = [i for i in range(y.shape[0])]
        if random:
            np.random.seed(seed)
            np.random.shuffle(idx)
        n1 = round(y.shape[0]*ratio)
        x1, y1 = x[idx[:n1]], y[idx[:n1]]
        x2, y2 = x[idx[n1:]], y[idx[n1:]]
        return (x1, y1), (x2, y2)

    
class DataFolderLoader(DataLoader):
    
    def __init__(self, root, batch_size, shuffle, tforms=[], ext='png', n_channels=3, seed=0):
        self.root = root
        self.fns = glob.glob(os.path.join(self.root, '**', '*.'+ext))
        random.seed(seed)
        random.shuffle(self.fns)
        
        label_str = list(set([os.path.split(os.path.split(fn)[0])[1] for fn in self.fns]))
        str2label = {k: i for i, k in enumerate(label_str)}
        labels = [str2label[os.path.split(os.path.split(fn)[0])[1]] for fn in self.fns]

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.dataset = tf.data.Dataset.from_tensor_slices((tf.constant(self.fns), labels))
        self.dataset = self.dataset.cache()
        if shuffle:
            self.dataset = self.dataset.shuffle(len(self.fns), reshuffle_each_iteration=True)
        if ext == 'png':
            self.dataset = self.dataset.map(PNGFile2LabeledExample(str2label, n_channels), num_parallel_calls=AUTOTUNE)
        else:
            assert(ext == 'jpg')
            self.dataset = self.dataset.map(JPGFile2LabeledExample(str2label, n_channels), num_parallel_calls=AUTOTUNE)

        for tform in tforms:
            self.dataset = self.dataset.map(tform, num_parallel_calls=AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(AUTOTUNE)

        
    def __len__(self):
        return len(self.fns)
    

    def __iter__(self):
        return self.dataset.__iter__()


    def __next__(self):
        return self.datset.__next__()
    
        
if __name__ == '__main__':
    ld = DataFolderLoader('Imagenet32/val', 100, True)

    for x, y in ld:
        print(x.shape, y.shape)
        print(y)

        

class MultiSourceDataset(DataLoader):
    def __init__(self, src_names, aug_params, batch_size, **kwargs):

        ## init source datasets
        ds_list = []
        for s, a in itertools.product(src_names, aug_params):
            if a is None:
                print("src: %s + none"%(s)) ##TODO: simplify
            else:
                print("src: %s + %s"%(s, " + ".join(a_param[0] for a_param in a)))
            ds_list.append(
                getattr(data, s)(
                    root=os.path.join('data', s.lower()),
                    batch_size=batch_size, # draw batch from each source
                    aug_list=a,
                    **kwargs)
            )
            
        ## convert to multi-source loaders
        self.train = tf.data.experimental.sample_from_datasets([d.train for d in ds_list])
        self.val = tf.data.experimental.sample_from_datasets([d.val for d in ds_list])
        self.test = tf.data.experimental.sample_from_datasets([d.test for d in ds_list])
        if 'domain_id' in kwargs.keys() and kwargs['domain_id'] is not None:
            self.train_dom = tf.data.experimental.sample_from_datasets([d.train_dom for d in ds_list])
            self.val_dom = tf.data.experimental.sample_from_datasets([d.val_dom for d in ds_list])
            self.test_dom = tf.data.experimental.sample_from_datasets([d.test_dom for d in ds_list])
            

# class MultiSourceDataset(DataLoader):
#     def __init__(self, ds_list, batch_size, train_shuffle=True, val_shuffle=False, test_shuffle=False, buffer_size=1000):

#         # self.train = self._init_loader(tf.data.experimental.sample_from_datasets([d.train for d in ds_list]), buffer_size, train_shuffle, batch_size, [Squeeze()])
#         # self.val = self._init_loader(tf.data.experimental.sample_from_datasets([d.val for d in ds_list]), buffer_size, val_shuffle, batch_size, [Squeeze()])
#         # self.test = self._init_loader(tf.data.experimental.sample_from_datasets([d.test for d in ds_list]), buffer_size, test_shuffle, batch_size, [Squeeze()])

#         # ##TODO
#         # sys.exit()
#         self.train = tf.data.experimental.sample_from_datasets([d.train for d in ds_list])
#         self.val = tf.data.experimental.sample_from_datasets([d.val for d in ds_list])
#         self.test = tf.data.experimental.sample_from_datasets([d.test for d in ds_list])


class JointLoader:
    def __init__(self, lds):
        self.lds = lds
        
    def __iter__(self):
        self.iters = [iter(ld) for ld in self.lds]
        self.iter_end = [False for ld in self.lds]
        return self
    
    def __next__(self):
        x_list, y_list = [], []
        for i, it in enumerate(self.iters):
            try:
                x, y = next(it)
            except StopIteration:
                self.iter_end[i] = True
                
                if all(self.iter_end):
                    raise StopIteration
                else:
                    self.iters[i] = iter(self.lds[i])
                    x, y = next(self.iters[i])
            x_list.append(x)
            y_list.append(y)
            
        # maintain the same batch size
        bs_min = min([o.shape[0] for o in x_list])
        x_list = [o[:bs_min] for o in x_list]
        x_list = tf.concat(x_list, 0)
        y_list = [o[:bs_min] for o in y_list]
        y_list = tf.concat(y_list, 0)
        return x_list, y_list
    

class ChainLoader:
    def __init__(self, ld1, ld2):
        self.ld1 = ld1
        self.ld2 = ld2
        
    def __iter__(self):
        self.iter = itertools.chain(self.ld1, self.ld2)
        return self
    
    def __next__(self):
        return next(self.iter)

    
class DomainDataset(DataLoader):
    def __init__(self, dsld_src, dsld_tar):
        self.train = JointLoader([dsld_src.train_dom, dsld_tar.train_dom])
        self.val = JointLoader([dsld_src.val_dom, dsld_tar.val_dom])
        self.test = JointLoader([dsld_src.test_dom, dsld_tar.test_dom])
        
        
class DomainDataset_old(DataLoader):
    def __init__(self, src_names, aug_params, tar, batch_size, buffer_size=1000,
                 train_shuffle=True, val_shuffle=False, test_shuffle=False,
                 train_aug=True, val_aug=False, test_aug=False,
                 **kwargs
    ):

        ## init source datasets
        src_ds_list = []
        for s, a in itertools.product(src_names, aug_params):
            if a is None:
                print("[dom] src: %s + none"%(s)) ##TODO: simplify
            else:
                print("[dom] src: %s + %s"%(s, " + ".join(a_param[0] for a_param in a)))
            src_ds_list.append(
                getattr(data, s)(
                    root=os.path.join('data', s.lower()),
                    batch_size=1, # draw one sample from each source
                    aug_list=a,
                    domain_id=1,
                    train_shuffle=train_shuffle, val_shuffle=val_shuffle, test_shuffle=test_shuffle,
                    train_aug=train_aug, val_aug=val_aug, test_aug=test_aug,
                    **kwargs
                )
            )

        ## init the target dataset
        tar_ds = getattr(data, tar)(
            root=os.path.join('data', tar.lower()),
            batch_size=1,
            domain_id=0,
            train_shuffle=train_shuffle, val_shuffle=val_shuffle, test_shuffle=test_shuffle,
            train_aug=False, val_aug=False, test_aug=False,
            **kwargs
        )
        
        ## init loaders
        train = [d.train_dom for d in src_ds_list] + [tar_ds.train_dom]
        val = [d.val_dom for d in src_ds_list] + [tar_ds.val_dom]
        test = [d.test_dom for d in src_ds_list] + [tar_ds.test_dom]
        self.train = self._init_loader(tf.data.experimental.sample_from_datasets(train),
                                       buffer_size, train_shuffle, batch_size, [Squeeze()])
        self.val = self._init_loader(tf.data.experimental.sample_from_datasets(val),
                                     buffer_size, val_shuffle, batch_size, [Squeeze()])
        self.test = self._init_loader(tf.data.experimental.sample_from_datasets(test),
                                      buffer_size, test_shuffle, batch_size, [Squeeze()])


    # class DomainDataset(DataLoader):
    # def __init__(self, src_list, tar, batch_size, train_shuffle=True, val_shuffle=False, test_shuffle=False, buffer_size=1000):
    #     raise NotImplementedError
    
    #     train = [d.train_dom for d in src_list] + [tar.train_dom]
    #     val = [d.val_dom for d in src_list] + [tar.val_dom]
    #     test = [d.test_dom for d in src_list] + [tar.test_dom]
    #     weights = [0.5/len(src_list)]*len(src_list) + [0.5]
        
    #     self.train = self._init_loader(tf.data.experimental.sample_from_datasets(train, weights=weights),
    #                                    buffer_size, train_shuffle, batch_size, [Squeeze()])
    #     self.val = self._init_loader(tf.data.experimental.sample_from_datasets(val, weights=weights),
    #                                  buffer_size, val_shuffle, batch_size, [Squeeze()])
    #     self.test = self._init_loader(tf.data.experimental.sample_from_datasets(test, weights=weights),
    #                                   buffer_size, test_shuffle, batch_size, [Squeeze()])




    
        
def rot_gaussian(rot, mu, cov):
    rot_rad = np.deg2rad(rot)
    R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
    mu_rot = np.transpose(np.matmul(R, np.transpose(mu)))
    cov_rot = np.matmul(np.matmul(R, cov), np.transpose(R))
    return mu_rot, cov_rot
    


"""
plot
"""
def plot_data(x, y, markers, colors, alphas, labels, facecolors=None, fn=None, markersize=2, linewidth=1, w=None, classifier=None):
    #y_id = np.unique(y)
    y_id = np.arange(0, y.max()+1)
    assert(len(y_id) == len(markers) == len(alphas))
    if facecolors is None:
        facecolors = colors
    plt.figure(1)
    plt.clf()

    ## plot data
    hs = []
    for y_i, m_i, c_i, fc_i, a_i, l_i in zip(y_id, markers, colors, facecolors, alphas, labels):
        x_i = x[y==y_i]
        h = plt.plot(x_i[:, 0], x_i[:, 1], m_i, alpha=a_i, markerfacecolor=fc_i, markeredgecolor=c_i, markersize=markersize, linewidth=linewidth, label=l_i)
        hs.append(h[0])

    ## plot decision
    if classifier is not None:
        X, Y = np.meshgrid(
            np.linspace(-2.5, 2.5, 100), 
            np.linspace(-2.5, 2.5, 100))
        XY = np.concatenate((X.flatten()[..., np.newaxis], Y.flatten()[..., np.newaxis]), 1)
        P = classifier(XY)
        Z = P[:, 1]
        Z = np.reshape(Z, X.shape)

        plt.contourf(X, Y, Z, alpha=0.3, cmap='RdYlGn')
        plt.colorbar()
        
        
    ## plot targetness
    if w is not None:
        X, Y = np.meshgrid(
            np.linspace(-2.5, 2.5, 100), 
            np.linspace(-2.5, 2.5, 100))
        XY = np.concatenate((X.flatten()[..., np.newaxis], Y.flatten()[..., np.newaxis]), 1)
        W = w(tf.constant(XY, dtype=tf.float32))
        W = np.reshape(W.numpy(), X.shape)
        Z = W / (1+W)

        Z = ((1 - 2*np.abs(Z-0.5)) > 1e-3).astype(np.float32)

        #fig, ax = plt.subplots(1,1)
        plt.contourf(X, Y, Z, alpha=0.5, zorder=5)
        plt.colorbar()
        
    ## beautify
    plt.grid('on')
    plt.gca().set_aspect('equal')
    plt.xlim((-2.5, 2.5))
    plt.ylim((-2.5, 2.5))
    plt.legend(handles=hs)
    if fn is not None:
        plt.savefig(fn+'.png', bbox_inches='tight')
        
    plt.close()

        
def plot_targetness(w, alpha=0.7, fn=None, fig_id=None, close=True):

    overlay = True if fig_id is not None else False

    X, Y = np.meshgrid(
        np.linspace(-2.5, 2.5, 100), 
        np.linspace(-2.5, 2.5, 100))
    XY = np.concatenate((X.flatten()[..., np.newaxis], Y.flatten()[..., np.newaxis]), 1)
    W = w(tf.constant(XY, dtype=tf.float32))
    W = np.reshape(W.numpy(), X.shape)
    Z = W / (1+W)

    if overlay:
        plt.figure(fig_id)
    else:
        plt.figure(1)
        plt.clf()
    fig, ax = plt.subplots(1,1)
    cp = ax.contourf(X, Y, Z, alpha=alpha, zorder=10)
    fig.colorbar(cp)
    if fn is not None:
        plt.savefig(fn+'.png')
    if close:
        plt.close()


def shuffle_labeled_examples(x, y, seed=None):
    assert(x.shape[0] == y.shape[0])
    n = y.shape[0]
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
        
    i_rnd = np.random.permutation(n)
    x, y = x[i_rnd], y[i_rnd]
    return x, y
