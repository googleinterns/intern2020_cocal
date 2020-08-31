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
    fig_root = 'snapshots/figs/src_%s_tar_%s%s'%(src_short, args.tar.lower(), '_aug' if aug else '')
    os.makedirs(fig_root, exist_ok=True)
    
    if aug:
        T_rng = np.arange(0.1, 10.0+0.1, 0.1)
        aug_params = [[ 
                ('intensity_flip', {}),
                ('intensity_scaling', {'min': -1.5, 'max': 1.5}),
                ('intensity_offset', {'min': -0.5, 'max': 0.5}),
                ('affine', {'std': 0.1}),
                ('translation', {'x_max': 2.0, 'y_max': 2.0}),
                ('gaussian', {'std': 0.1}),
            ]]
    else:
        T_rng = np.concatenate((np.array([1.0]), np.arange(10.0, 500.0+10.0, 10.0)))
        aug_params = [None]
    print(T_rng)


        
    ## init a loader
    ds_src = data.MultiSourceDataset(args.src, aug_params, batch_size=100, val_shuffle=True, val_aug=True, test_aug=True)
    ds_tar = getattr(data, 'SVHN')(batch_size=100, val_shuffle=True)

    ## load a trained model
    mdl = getattr(model, 'ResNet18')(num_class=10, activation='relu', input_shape=(32, 32, 3)) ##TODO: generalize
    mdl_cal = model.TempCls(mdl)
    mdl_cal.load_weights('snapshots/%s2svhn%s_ResNet18/model_params_cal_best'%(src_short, '_aug' if aug else ''))

    
    ## error/ece
    Ts, eces_src, eces_tar = [], [], []
    for T in T_rng:
        mdl_cal.T = tf.constant(T, dtype=tf.float32)
        Ts.append(T)

        for ds_type in ['src', 'tar']:
            dsld = ds_tar.test if ds_type == 'tar' else ds_src.test
            
            ## target
            ph_list, yh_list, y_list = [], [], []

            for x, y in dsld:
                logits = mdl_cal(x, training=False)['logits']
                ph = tf.math.reduce_max(tf.nn.softmax(logits, -1), -1)
                yh = tf.math.argmax(logits, -1)
                ph_list.append(ph.numpy())
                yh_list.append(yh.numpy())
                y_list.append(y.numpy())

            ece = ECE(np.concatenate(ph_list), np.concatenate(yh_list), np.concatenate(y_list),
                      rel_diag_fn=os.path.join(fig_root, 'rel_diag_%s%s_T_%f'%(src_short, '_aug' if aug else '', T)))
            error = np.mean(np.concatenate(y_list) != np.concatenate(yh_list))

            if ds_type == 'src':
                ece_src = ece
                error_src = error
                eces_src.append(ece_src*100.0)

                ## draw the best reliability diagram
                if T == Ts[np.argmin(np.array(eces_src))]:
                    ECE(np.concatenate(ph_list), np.concatenate(yh_list), np.concatenate(y_list),
                        rel_diag_fn=os.path.join(fig_root, 'rel_diag_%s%s_src_best_T'%(src_short, '_aug' if aug else '')))

            else:
                ece_tar = ece
                error_tar = error
                eces_tar.append(ece_tar*100.0)

                ## draw the best reliability diagram
                if T == Ts[np.argmin(np.array(eces_tar))]:
                    ECE(np.concatenate(ph_list), np.concatenate(yh_list), np.concatenate(y_list),
                        rel_diag_fn=os.path.join(fig_root, 'rel_diag_%s%s_tar_best_T'%(src_short, '_aug' if aug else '')))

            
        print("T = %f, error_src = %f, error_tar = %f, ECE_src = %.2f%%, ECE_tar = %.2f%%"%(
            T,
            error_src, error_tar, 
            ece_src*100.0, ece_tar*100.0))

        
        ## plot
        plt.figure(1)
        plt.clf()

        h1 = plt.plot(Ts, eces_tar, 'r-', label='target')
        h2 = plt.plot(Ts, eces_src, 'b-', label='source')

        plt.plot(Ts[np.argmin(np.array(eces_tar))], min(eces_tar), 'rs')
        plt.plot(Ts[np.argmin(np.array(eces_src))], min(eces_src), 'bs')

        plt.xlabel('temperature', fontsize=fontsize)
        plt.ylabel('ECE (%%)', fontsize=fontsize)
        plt.grid('on')
        plt.legend(handles=[h2[0], h1[0]], fontsize=fontsize)

        ## save
        plt.savefig(os.path.join(fig_root, 'plot_temp_trend_%s%s.png'%(src_short, '_aug' if aug else '')), bbox_inches='tight')
        plt.close()
    
    


