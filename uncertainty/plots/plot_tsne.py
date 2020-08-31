import os, sys
import numpy as np
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf

import data
import model 

from MulticoreTSNE import MulticoreTSNE as TSNE


def plot_tsne(ld, ld_name, mdl):
    ## load features
    feats = []
    y_list = []
    for x, y in ld:
        feat = mdl(x, training=False)['embeds']
        feats.append(feat)
        y_list.append(y)
    feats = tf.concat(feats, 0)
    y_list = tf.concat(y_list, 0)
    print(feats.shape)

    embeddings = TSNE(n_jobs=10).fit_transform(feats)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    plt.figure(1)
    plt.clf()
    plt.scatter(vis_x, vis_y, c=y_list, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig('tsne_%s.png'%(ld_name), bbox_inches='tight')
    plt.close()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plot ECE trend')

    ## meta args
    parser.add_argument('--src', type=str, nargs='*', help='list of sources') ##TODO: how to restrict possible inputs?
    parser.add_argument('--tar', type=str, default='SVHN')

    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--fontsize', type=int, default=15)
    
    args = parser.parse_args()
    fontsize = args.fontsize
    aug = args.aug
    src_short = ''
    for n in args.src:
        src_short += n[0].lower()
    fig_root = 'snapshots/figs/src_%s_tar_%s%s'%(src_short, args.tar.lower(), '_aug' if aug else '')
    os.makedirs(fig_root, exist_ok=True)
    
    if aug:
        aug_params = [[ 
                ('intensity_flip', {}),
                ('intensity_scaling', {'min': -1.5, 'max': 1.5}),
                ('intensity_offset', {'min': -0.5, 'max': 0.5}),
                ('affine', {'std': 0.1}),
                ('translation', {'x_max': 2.0, 'y_max': 2.0}),
                ('gaussian', {'std': 0.1}),
            ]]
    else:
        aug_params = [None]


        
    ## init a loader
    ds_src = data.MultiSourceDataset(args.src, aug_params, batch_size=100, val_shuffle=True, val_aug=True, test_aug=True)
    ds_tar = getattr(data, 'SVHN')(batch_size=100, val_shuffle=True)

    ## load a trained model
    mdl = getattr(model, 'ResNet18')(num_class=10, activation='relu', input_shape=(32, 32, 3)) ##TODO: generalize
    mdl.load_weights('snapshots/%s2svhn%s_ResNet18/model_params_best'%(src_short, '_aug' if aug else ''))


    ## plot
    plot_tsne(ds_src.val, "src", mdl)
    plot_tsne(ds_tar.val, "tar", mdl)
    
    sys.exit()

    ## load source features
    feats = []
    y_list = []
    for x, y in ds_src.val:
        feat = mdl(x, training=False)['embeds']
        feats.append(feat)
        y_list.append(y)
    feats = tf.concat(feats, 0)
    y_list = tf.concat(y_list, 0)
    print(feats.shape)

    embeddings = TSNE(n_jobs=10).fit_transform(feats)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    plt.figure(1)
    plt.clf()
    plt.scatter(vis_x, vis_y, c=y_list, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.savefig('tsne_src.png', bbox_inches='tight')
    plt.close()
    
    sys.exit()

    
