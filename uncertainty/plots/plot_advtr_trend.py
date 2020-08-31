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
from learning import LearnerDACls


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plot advtr trend')

    ## args
    parser.add_argument('--src', type=str, nargs='*', help='list of sources') ##TODO: how to restrict possible inputs?
    parser.add_argument('--tar', type=str, default='SVHN')
    parser.add_argument('--n_labels', type=int, default=10)
    parser.add_argument('--best_model', action='store_true', help='load the best model')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--img_size', type=int, nargs=3, default=(32, 32, 3), help='image size')
    parser.add_argument('--model', default='ResNet18',  type=str, help='model name')
    parser.add_argument('--model_adv', default='MidAdvFNN',  type=str, help='model name')
    parser.add_argument('--fontsize', type=int, default=15)
    parser.add_argument('--log_scale', action='store_true')
    parser.add_argument('--smalln', action='store_true')
    parser.add_argument('--aug', action='store_true')
    
    args = parser.parse_args()

    src_short = "".join([s[0] for s in args.src])

    fig_root = os.path.join('snapshots', 'figs', 'advtr_%s2%s'%(src_short[0].lower(), args.tar[0].lower()))
    if args.log_scale:
        reg_param_adv_list = ['0.001', '0.01', '0.1', '1.0', '10.0', '100.0']
    else:
        reg_param_adv_list = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

    os.makedirs(fig_root, exist_ok=True)
    
    ## init datasets
    print("## init datasets")
    # ds_src = data.MultiSourceDataset(args.src, [None], batch_size=args.batch_size, val_shuffle=True, val_aug=True, domain_id=1,
    #                                  color=False if args.img_size[2]==1 else True, size=args.img_size[0])
    ds_tar = getattr(data, args.tar)(root=os.path.join('data', args.tar.lower()), batch_size=args.batch_size, val_shuffle=True, domain_id=0,
                                     color=False if args.img_size[2]==1 else True, size=args.img_size[0])


    ## gather results
    error_list, ece_list, ent_list = [], [], []
    for reg in reg_param_adv_list:
        ## init a base model
        mdl_base = getattr(model, args.model)(num_class=args.n_labels, activation='relu', input_shape=args.img_size) ##TODO: generalize
        mdl_adv = getattr(model, args.model_adv)(n_in=800)#mdl_base.dim_feat) 
        mdl = model.DAN(mdl_base, mdl_adv)
        if args.best_model:
            mdl.load_weights('snapshots/%s2%s_advtr_%s%s%s_%s/model_params_best'%(src_short[0].lower(), args.tar[0].lower(), reg.replace('.', 'd'),
                                                                                '_smalln' if args.smalln else '',
                                                                                '_aug' if args.aug else '', 
                                                                                args.model))
        else:
            mdl.load_weights('snapshots/%s2%s_advtr_%s%s%s_%s/model_params_final'%(src_short[0].lower(), args.tar[0].lower(), reg.replace('.', 'd'),
                                                                                   '_smalln' if args.smalln else '',
                                                                                   '_aug' if args.aug else '', 
                                                                                   args.model))

        ## compute error and ece
        learner = LearnerDACls(None, mdl)
        error, ece, ece_oc, ent = learner.test(ds_tar.test, ld_name=args.tar)
        error_list.append(error.numpy())
        ece_list.append(ece)
        ent_list.append(ent.numpy())

        print("reg = ", reg)
        print("error :", error_list)
        print("ece :", ece_list)
        print("ent :", ent_list)

        
    ## plot accuracy
    plt.figure(1)
    plt.clf()

    plt.plot(error_list, ece_list,  'ks-')

    plt.grid('on')
    plt.xlabel('error', fontsize=args.fontsize)
    plt.ylabel('ECE', fontsize=args.fontsize)

    plt.savefig(os.path.join(fig_root, 'plot_error_ece_%s%s%s%s.png'%(
        'best' if args.best_model else 'final',
        '_log' if args.log_scale else '',
        '_smalln' if args.smalln else '',
        '_aug' if args.aug else ''
    )), bbox_inches='tight')
    plt.close()

    ## plot entropy
    plt.figure(1)
    plt.clf()

    plt.plot(error_list, ent_list, 'ks-')

    plt.grid('on')
    plt.xlabel('error', fontsize=args.fontsize)
    plt.ylabel('entropy', fontsize=args.fontsize)

    plt.savefig(os.path.join(fig_root, 'plot_error_entropy_%s%s%s%s.png'%(
        'best' if args.best_model else 'final',
        '_log' if args.log_scale else '',
        '_smalln' if args.smalln else '',
        '_aug' if args.aug else ''
    )), bbox_inches='tight')
    plt.close()

    


             
    



                    
        

