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
from uncertainty import ECE


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
    fig_root = 'snapshots/figs/src_%s_tar_%s%s/imgs_iw'%(src_short, args.tar.lower(), '_aug' if aug else '')
    os.makedirs(fig_root, exist_ok=True)
    
    if aug:
        ##TODO: assume only one aug
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
    #ds_src = data.MultiSourceDataset(args.src, aug_params, batch_size=100, val_shuffle=True, val_aug=True, test_aug=True)
    ds_src = [getattr(data, s)(root=os.path.join('data', s.lower()), batch_size=100, aug_list=aug_params[0], val_shuffle=True, val_aug=True, test_aug=True) for s in args.src] 
    #ds_tar = getattr(data, 'SVHN')(batch_size=100, val_shuffle=True)

    ## load a trained model
    mdl = getattr(model, 'ResNet18')(num_class=10, activation='relu', input_shape=(32, 32, 3)) ##TODO: generalize
    mdl_cal = model.TempCls(mdl)
    mdl_cal.load_weights('snapshots/%s2svhn%s_ResNet18/model_params_cal_best'%(src_short, '_aug' if aug else ''))

    ## load a source-discriminator model
    mdl_sd = model.SourceDisc(mdl=mdl)
    mdl_sd_cal = model.TempCls(mdl_sd)
    mdl_sd_cal.load_weights('snapshots/%s2svhn%s_ResNet18/model_params_iw_cal_best'%(src_short, '_aug' if aug else ''))

    ## compute sd
    prob_src = []
    i = 0
    for name_src, ld_src in zip(args.src, ds_src):
        prob_src_i = []
        for x, _ in ld_src.test:
            logits = mdl_sd_cal(x, training=False)['logits']
            ph = tf.nn.softmax(logits, -1)
            ph = ph[:, 1]
            prob_src_i.append(ph.numpy())
            for x_tar in x[ph<0.05]:
                # normalize the intensity scale for visualization
                #x_tar = x_tar - tf.math.reduce_min(x_tar)
                #x_tar = x_tar / tf.math.reduce_max(x_tar)
                x_tar = tf.maximum(tf.minimum(x_tar, 1.0), 0.0)
                tf.io.write_file(
                    os.path.join(fig_root, "%s_tar_%d.png"%(name_src, i)),
                    tf.image.encode_png(tf.cast(x_tar*255, tf.uint8)))
                i += 1

            for x_tar in x[ph>0.95]:
                # normalize the intensity scale for visualization
                #x_tar = x_tar - tf.math.reduce_min(x_tar)
                #x_tar = x_tar / tf.math.reduce_max(x_tar)
                x_tar = tf.maximum(tf.minimum(x_tar, 1.0), 0.0)
                print(x_tar.numpy().min(), x_tar.numpy().max())
                tf.io.write_file(
                    os.path.join(fig_root, "%s_src_%d.png"%(name_src, i)),
                    tf.image.encode_png(tf.cast(x_tar*255, tf.uint8)))
                i += 1

                
        prob_src.append(np.concatenate(prob_src_i))

    ## box plot
    plt.figure(1)
    plt.clf()

    plt.boxplot(prob_src, whis=0.0, boxprops=dict(linewidth=3.0), medianprops=dict(linewidth=3.0), flierprops=dict(markersize=1))
    
    plt.gca().set_xticklabels(args.src, fontsize=fontsize)
    plt.ylabel('source probability', fontsize=fontsize)
    plt.grid('on')
    plt.ylim((0.0, 1.))
    
    ## save
    plt.savefig(os.path.join(fig_root, 'iw_trend.png'), bbox_inches='tight')
    plt.close()
    
    
