import os, sys
import numpy as np
import argparse
import glob
import types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf

import data
import model 
from util import *


def get_results_self(args, ds_src, ds_tar, ideal=False):
    from learning import LearnerClsSelf
    from learning import TempScalingCls as CalibratorCls

    #exp_name_list = glob.glob('snapshots/m2m_self_small_?')
    exp_name_list = [
        'snapshots/m2m_self_small%s_0'%('_ideal' if ideal else ''),
        'snapshots/m2m_self_small%s_1'%('_ideal' if ideal else ''),
        'snapshots/m2m_self_small%s_2'%('_ideal' if ideal else ''),
        'snapshots/m2m_self_small%s_3'%('_ideal' if ideal else ''),
        #'snapshots/m2m_self_small_4',
        'snapshots/m2m_self_small%s_5'%('_ideal' if ideal else ''),
        'snapshots/m2m_self_small%s_6'%('_ideal' if ideal else ''),
    ]
    
    error_cls_list, error_cal_list = [], []
    for exp_name in exp_name_list:
        params = types.SimpleNamespace()
        params.save_root = exp_name
        params.find_best = False
        params.load_final = False
        print(params.save_root)
        
        ## init a base model
        mdl_st_base = getattr(model, args.model.base)(num_class=args.data.n_labels, input_shape=args.data.img_size)
        mdl_st_base = model.TempCls(mdl_st_base)
        mdl_st = model.Student(args.model, mdl_st_base, ds_src, ds_tar, ideal=ideal)

        #mdl_tc_base = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
        #mdl_tc_base = model.TempCls(mdl_tc_base)
        mdl_tc_base = mdl_st_base #shared model
        mdl_tc = model.Teacher(args.model, mdl_tc_base, ds_src, ds_tar, ideal=ideal)

        ## init a learner
        learner = LearnerClsSelf(params, None, None, mdl_st, mdl_tc, ideal=ideal)
        ## train the model
        learner.train(ds_src, None, None) ##TODO: teacher model contains the loaders
        ## test the model
        error_cls, error_cal, *_ = learner.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
    
        error_cls_list.append(error_cls.numpy())
        error_cal_list.append(error_cal)

    return np.array(error_cls_list), np.array(error_cal_list)


def get_results_sourceonly(args, ds_src, ds_tar):
    from learning import LearnerCls
    from learning import TempScalingCls as CalibratorCls

    exp_name_list = glob.glob('snapshots/m2m_srconly_small_*')

    error_cls_list, error_cal_list = [], []
    for exp_name in exp_name_list:
        params = types.SimpleNamespace()
        params.save_root = exp_name
        params.find_best = False
        params.load_final = False
        print(params.save_root)
        
        ## init a base model
        mdl = getattr(model, args.model.base)(num_class=args.data.n_labels, input_shape=args.data.img_size)
        ## init a learner
        learner = LearnerCls(params, mdl)
        ## train the model
        learner.train(ds_src.train, None, None)
        ## test the model
        error_cls, error_cal, _ = learner.test(ds_tar.test, ld_name=args.data.tar, verbose=True)

        error_cls_list.append(error_cls.numpy())
        error_cal_list.append(error_cal)

    return np.array(error_cls_list), np.array(error_cal_list)
    
                           
def get_results_advtr(args, ds_src, ds_tar):
    from learning import LearnerDACls
    from learning import TempScalingCls as CalibratorCls

    exp_name_list = glob.glob('snapshots/m2m_advtr_small_*')

    error_cls_list, error_cal_list = [], []
    for exp_name in exp_name_list:
        params = types.SimpleNamespace()
        params.save_root = exp_name
        params.find_best = False
        params.load_final = False
        params.advtr_type = 'DANN'
        params.schedule_reg_param_adv = True
        print(params.save_root)

        ## init a base model
        mdl = getattr(model, args.model.base)(num_class=args.data.n_labels, input_shape=args.data.img_size)
        ## init an adv model
        mdl_adv = getattr(model, 'BigAdvFNN')(n_in=mdl.dim_feat)
        ## init a learner
        learner = LearnerDACls(params, model.DAN(mdl, mdl_adv))
        ## train the model
        learner.train([ds_src.train, ds_dom.train], None, None)
        ## test the model
        error_cls, error_cal, *_ = learner.test(ds_tar.test, ld_name=args.data.tar, verbose=True)

        error_cls_list.append(error_cls.numpy())
        error_cal_list.append(error_cal)

    return np.array(error_cls_list), np.array(error_cal_list)
    



if __name__ == '__main__':

    ## inint a parser
    parser = argparse.ArgumentParser(description='digit dataset training')

    ## dataset args
    parser.add_argument('--data.batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--data.n_labels', default=10, type=int, help='the number of labels')
    parser.add_argument('--data.src', type=str, nargs='*', default=['MNIST'], help='list of sources')
    parser.add_argument('--data.tar', type=str, default='MNIST', help='target') 
    parser.add_argument('--data.aug', type=str, nargs='*', default=[''], help='list of data augmentation')
    parser.add_argument('--data.img_size', type=int, nargs=3, default=(32, 32, 3), help='image size')
    parser.add_argument('--data.sample_ratio', type=float, nargs=2, default=[0.1, 0.1])
    
    parser.add_argument('--model.base', default='ResNet18', type=str, help='model name')
    parser.add_argument('--model.conf', default='ConfPred', type=str, help='model name')

    args = parser.parse_args()
    args = to_tree_namespace(args)
    fontsize = 15
    fig_root = 'snapshots/figs/plot_why_self'
    os.makedirs(fig_root, exist_ok=True)
    
    ## init aug parameters
    args.aug_params = []
    for a in args.data.aug:
        if a == 'jitter':
            args.aug_params.append([('jitter', {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4})])
            
        elif a == 'shake':
            args.aug_params.append([('randaug', {'size': 32, 'mode': 'SHAKE'})])
            
        elif a == 'svhnspec':
            args.aug_params.append([ 
                ('intensity_flip', {}),
                ('intensity_scaling', {'min': -1.5, 'max': 1.5}),
                ('intensity_offset', {'min': -0.5, 'max': 0.5}),
                ('affine', {'std': 0.1}),
                ('translation', {'x_max': 2.0, 'y_max': 2.0}),
                ('gaussian', {'std': 0.1}),
            ])
        else:
            ##TODO: simplify
            args.aug_params.append(None)

    
    ## init datasets
    print("## init datasets")
    ds_src = data.MultiSourceDataset(args.data.src, args.aug_params, batch_size=args.data.batch_size, val_shuffle=True, val_aug=True, domain_id=1,
                                     color=False if args.data.img_size[2]==1 else True, size=args.data.img_size[0], sample_ratio=args.data.sample_ratio[0])
    ds_tar = getattr(data, args.data.tar)(root=os.path.join('data', args.data.tar.lower()), batch_size=args.data.batch_size, val_shuffle=True, domain_id=0,
                                     color=False if args.data.img_size[2]==1 else True, size=args.data.img_size[0], sample_ratio=args.data.sample_ratio[1])
    ds_dom = data.DomainDataset(ds_src, ds_tar)

    ## load results
    [error_cls_self_ideal, error_cal_self_ideal] = get_results_self(args, ds_src, ds_tar, ideal=True)

    [error_cls_self, error_cal_self] = get_results_self(args, ds_src, ds_tar)

    [error_cls_srconly, error_cal_srconly] = get_results_sourceonly(args, ds_src, ds_tar)
    [error_cls_advtr, error_cal_advtr] = get_results_advtr(args, ds_src, ds_tar)
    

    ## plot
    plt.figure(1)
    plt.clf()

    h_srconly = plt.plot(error_cls_srconly*100.0, error_cal_srconly*100.0, 'kx', alpha=0.5, label='source-only')
    h_advtr = plt.plot(error_cls_advtr*100.0, error_cal_advtr*100.0, 'rx', alpha=0.5, label='adversarial')
    h_self = plt.plot(error_cls_self*100.0, error_cal_self*100.0, 'gx', alpha=0.5, label='self')
    h_self_ideal = plt.plot(error_cls_self_ideal*100.0, error_cal_self_ideal*100.0, 'bx', alpha=0.5, label='self-ideal')

    h_srconly_mean = plt.plot(np.mean(error_cls_srconly)*100.0, np.mean(error_cal_srconly)*100.0, 'ks', label='source-only mean')
    h_advtr_mean = plt.plot(np.mean(error_cls_advtr)*100.0, np.mean(error_cal_advtr)*100.0, 'rs', label='adversarial mean')
    h_self_mean = plt.plot(np.mean(error_cls_self)*100.0, np.mean(error_cal_self)*100.0, 'gs', label='self mean')
    h_self_ideal_mean = plt.plot(np.mean(error_cls_self_ideal)*100.0, np.mean(error_cal_self_ideal)*100.0, 'bs', label='self-ideal mean')

    plt.grid('on')
    plt.xlabel('classification error (%)', fontsize=fontsize)
    plt.ylabel('ECE (%)', fontsize=fontsize)
    plt.legend(handles=[h_srconly[0], h_advtr[0], h_self[0], h_self_ideal[0]], fontsize=fontsize, loc='upper right')

    
    plt.savefig(os.path.join(fig_root, 'exp_iid.png'), bbox_inches='tight')
    plt.close()

    
    
