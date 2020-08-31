import os, sys
import argparse
import types
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf

import data
import model 
from util import *
from learning import LearnerCls, LearnerDACls, LearnerClsRT, LearnerConfPred
from learning import TempScalingCls as CalibratorCls

##TODO: clean-up tf options
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

def plot_prec_cov(T, prec, cov, fn, fontsize=15):

    plt.figure(1)
    plt.clf()

    plt.plot(cov, prec, 'rs-')
    plt.xlabel('coverage (%)', fontsize=fontsize)
    plt.ylabel('precision (%)', fontsize=fontsize)
    plt.grid('on')
    plt.savefig(fn+'.png', bbox_inches='tight')


def main(args):

    data_fn = 'plots/prec_cov_list_twoparams.pk'
    fig_fn = 'plots/prec_cov_twoparams'
    if os.path.exists(data_fn):
        pc_data = pickle.load(open(data_fn, 'rb'))
        plot_prec_cov(pc_data['T_list'], pc_data['prec_list'], pc_data['cov_list'], fig_fn)
        return

    ## init a snapshot path
    os.makedirs(args.train.save_root, exist_ok=True)

    ## init logger
    sys.stdout = Logger(os.path.join(args.train.save_root, 'out'))

    ## print args
    print_args(args)

    ## init gpus
    if not args.cpu:
        print("##GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print()

    ## init datasets
    print("## init datasets")
    ds_src = data.MultiSourceDataset(
        args.data.src,
        args.aug_params,
        batch_size=args.data.batch_size,
        val_shuffle=True,
        val_aug=True,
        domain_id=1,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[0])
    assert(len(args.aug_params) == 1) ##TODO
    ds_tar = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        aug_list=args.aug_params[0],
        val_shuffle=True,
        val_aug=True,
        domain_id=0,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[1])
    ds_dom = data.DomainDataset(
        data.MultiSourceDataset(
            args.data.src,
            args.aug_params,
            batch_size=args.data.batch_size,
            val_shuffle=True,
            val_aug=True,
            test_aug=True, #diff
            domain_id=1,
            color=False if args.data.img_size[2]==1 else True,
            size=args.data.img_size[0],
            sample_ratio=args.data.sample_ratio[0]),
        getattr(data, args.data.tar)(
            root=os.path.join('data', args.data.tar.lower()),
            batch_size=args.data.batch_size,
            aug_list=args.aug_params[0],
            val_shuffle=True,
            val_aug=True,
            test_aug=True, #diff
            domain_id=0,
            color=False if args.data.img_size[2]==1 else True,
            size=args.data.img_size[0],
            sample_ratio=args.data.sample_ratio[1]))
    print()    

    ####
    ## reliable teacher learning
    ####    
    mdl_st_base = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
    #mdl_st_base = model.TempCls(mdl_st_base)
    mdl_st = model.Student(args.model, mdl_st_base, ds_src, ds_tar, ideal=args.ideal)

    mdl_tc_base = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
    #mdl_tc_base = model.TempCls(mdl_tc_base)
    mdl_tc = model.Teacher(args.model, mdl_tc_base, ds_src, ds_tar, ideal=args.ideal)


    ## rename
    model_t = mdl_tc
    model_s = mdl_st
    model_c = model_s.model_base
    params = args.train
    params_base = args.train_base
    params_advtr = args.train_advtr
    params_iw = args.train_iw
    params_iw_cal = args.cal_iw
    params_conf = args.est_conf
    i_epoch = 1
    
    ## init a model
    if params.init == 'sourceonly':

        ##TODO: assume classification
        print("## init the student model with sourceonly training")
        model.set_trainable(model_c, True)
        ## init a learner
        learner = LearnerCls(params_base, model_c, model_name_postfix='_sourceonlyinit')
        ## train the model
        learner.train(ds_src.train, ds_src.val, ds_src.test)
        ## test the model
        learner.test(ds_src.test, ld_name='src', verbose=True)
        print()

    elif params.init == 'advtr':
        ##TODO: assume classification
        print("## init a base model with adversarial training")
        model.set_trainable(model_c, True)
        ## init a adv model
        mdl_adv = getattr(model, params_advtr.model_advtr)(n_in=model_c.dim_feat)
        ## init a learner
        learner = LearnerDACls(params_advtr, model.DAN(model_c, mdl_adv), model_name_postfix='_advtrinit')
        ## train the model
        learner.train([ds_src.train, ds_dom.train], None, ds_tar.test)        
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        print()            

    else:
        raise NotImplementedError

    ## init iw
    if model_t.train.model_conf.model_iw is not None:
        print("## learn IW")
        model_sd = model_t.train.model_conf.model_iw.model_sd.model
        model_sd.train()

        ## init a learner
        learner_sd = LearnerCls(params_iw, model_sd, model_name_postfix='_iw_epoch_%d'%(i_epoch))
        ## train the model
        learner_sd.train(ds_dom.train, ds_dom.val, ds_dom.test)
        ## test the model
        learner_sd.test(ds_dom.test, ld_name='domain', verbose=True)
        print()

        ## init a calibraton model
        model_sd_cal = model_t.train.model_conf.model_iw.model_sd
        model_sd_cal.train()

        ## init a calibrator
        calibrator_iw = CalibratorCls(params_iw_cal, model_sd_cal, model_name_postfix='_iw_cal_epoch_%d'%(i_epoch))
        ## calibrate the model
        calibrator_iw.train(ds_dom.val, ds_dom.val, ds_dom.test)
        ## test the model
        calibrator_iw.test(ds_dom.test, ld_name='domain', verbose=True)
        print()

        ## 2. learn confidence predictor
        model_base = model_t.train.model_base
        #model_conf = model_t.train.model_conf
        model_iw = model_t.train.model_conf.model_iw
        #model_iw_cond = model.CondIW(model_iw, model_conf, ds_src.train, ds_tar.train)
        model_conf = model.TwoParamsConfPred(model_base, model_iw)
        
        
        ## init a learner
        learner = LearnerConfPred(params_conf, model_conf, model_base, None, model_name_postfix='_confpred_epoch_%d'%(i_epoch))
        # ## train the model
        # learner.train(ds_src.val, ds_src.val, ds_tar.test)
        # ## test the model
        # learner.test(ds_tar.test, ld_name='tar', verbose=True)
        # learner.test(ds_tar.train, ld_name='tar (train)', verbose=True)
        # print()
        
    else:
        model_base = model_t.train.model_base
        model_conf = model_t.train.model_conf

        ## init a learner
        learner = LearnerConfPred(params_conf, model_conf, model_base, None, model_name_postfix='_confpred_epoch_%d'%(i_epoch))
        ## train the model
        model_conf.T = tf.Variable(1.0 - params_conf.eps) ##TODO
        print("T = %f"%(model_conf.T.numpy()))
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        learner.test(ds_tar.train, ld_name='tar (train)', verbose=True)

    ## compute precision and coverage
    T_list, prec_list, cov_list = [], [], []
    rng = [0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.8, 0.7, 0.6, 0.5]
    for T_bnd in rng:
        for T_iw in rng:
            model_conf.T_bnd = T_bnd
            model_conf.T_iw = T_iw
            prec, n_conf, n = learner.test(ds_tar.train, ld_name='tar (train)', verbose=True)
            T_list.append((T_bnd, T_iw))
            prec_list.append(prec.numpy())
            cov_list.append(float(n_conf)/float(n))

            print(T_list)
            print(prec_list)
            print(cov_list)
            print()

    T_list = np.array(T_list)
    prec_list = np.array(prec_list)
    cov_list = np.array(cov_list)
    pickle.dump({'T_list': T_list, 'prec_list': prec_list, 'cov_list': cov_list}, open(data_fn, 'wb'))
    

    
def parse_args():
    ## inint a parser
    parser = argparse.ArgumentParser(description='digit dataset training')

    ## meta args
    parser.add_argument('--exp_name', required=True, type=str, help='experiment name')
    parser.add_argument('--snapshot_root', default='snapshots', type=str, help='snapshot root name')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--ideal', action='store_true', help='enable cheatkey')

    ## dataset args
    parser.add_argument('--data.batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--data.n_labels', default=10, type=int, help='the number of labels')
    parser.add_argument('--data.src', type=str, nargs='*', default=['MNIST'], help='list of sources')
    parser.add_argument('--data.tar', type=str, default='USPS', help='target') 
    parser.add_argument('--data.aug', type=str, nargs='*', default=[''], help='list of data augmentation')
    parser.add_argument('--data.img_size', type=int, nargs=3, default=(32, 32, 3), help='image size')
    parser.add_argument('--data.sample_ratio', type=float, nargs=2, default=[1.0, 1.0])
    
    ## model args
    parser.add_argument('--model.base', default='ResNet18', type=str, help='model name')
    parser.add_argument('--model.iw', default='BigFNN', type=str, help='model name')
    parser.add_argument('--model.conf', default='ConfPred', type=str, help='model name')

    ## RT train args
    parser.add_argument('--train.find_best', action='store_true', help='find the best model')
    parser.add_argument('--train.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train.n_epochs', type=int, default=5, help='the number of training iterations')
    parser.add_argument('--train.init', type=str, default='advtr', help='model initialization approach')
    parser.add_argument('--train.val_period', default=1, type=int, help='validation period in epochs')
    
    ## base model train args
    parser.add_argument('--train_base.find_best', action='store_true', help='find the best model')
    parser.add_argument('--train_base.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train_base.optim', default='SGD', type=str, help='optimizer')
    parser.add_argument('--train_base.lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--train_base.lr_step_size', default=20, type=float, help='stepsize for step learning rate scheduler')
    parser.add_argument('--train_base.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    parser.add_argument('--train_base.weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('--train_base.momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--train_base.n_epochs', default=100, type=int, help='the number of epochs')
    parser.add_argument('--train_base.val_period', default=1, type=int, help='validation period in epochs')

    ## iw train args
    parser.add_argument('--train_iw.find_best', action='store_true', help='find the best model')
    parser.add_argument('--train_iw.load_final', action='store_true', help='load the final model')    
    parser.add_argument('--train_iw.optim', default='SGD', type=str, help='optimizer')
    parser.add_argument('--train_iw.lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--train_iw.lr_step_size', default=20, type=float, help='stepsize for step learning rate scheduler')
    parser.add_argument('--train_iw.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    parser.add_argument('--train_iw.weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('--train_iw.momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--train_iw.n_epochs', default=100, type=int, help='the number of epochs')
    parser.add_argument('--train_iw.val_period', default=1, type=int, help='validation period in epochs')

    ## cal args
    parser.add_argument('--cal_iw.find_best', action='store_true', help='find the best model')
    parser.add_argument('--cal_iw.load_final', action='store_true', help='load the final model')    
    parser.add_argument('--cal_iw.optim', default='SGD', type=str, help='optimizer')
    parser.add_argument('--cal_iw.lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--cal_iw.lr_step_size', default=50, type=float, help='stepsize for step learning rate scheduler')
    parser.add_argument('--cal_iw.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    parser.add_argument('--cal_iw.weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('--cal_iw.momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--cal_iw.n_epochs', default=500, type=int, help='the number of epochs')
    parser.add_argument('--cal_iw.val_period', default=1, type=int, help='validation period in epochs')

    ## train args
    parser.add_argument('--train_advtr.find_best', action='store_true', help='find the best model')
    parser.add_argument('--train_advtr.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train_advtr.optim', default='SGD', type=str, help='optimizer')
    parser.add_argument('--train_advtr.lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--train_advtr.lr_step_size', default=20, type=float, help='stepsize for step learning rate scheduler')
    parser.add_argument('--train_advtr.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    parser.add_argument('--train_advtr.weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('--train_advtr.momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--train_advtr.n_epochs', default=100, type=int, help='the number of epochs')
    parser.add_argument('--train_advtr.val_period', default=1, type=int, help='validation period in epochs')
    parser.add_argument('--train_advtr.advtr_type', type=str, default='DANN', help='domain-adversarial training type')

    parser.add_argument('--train_advtr.model_advtr', type=str, default='BigAdvFNN', help='adversarial network name')
    parser.add_argument('--train_advtr.reg_param_adv', type=float, default=1.0, help='adversarial loss regularization parameter')
    #parser.add_argument('--train_advtr.schedule_reg_param_adv', action='store_true', help='schedule the adversarial loss regularization parameter')
    parser.add_argument('--train_advtr.no_adv_reg_schedule', action='store_true', help='do not schedule the adversarial loss regularization parameter')

    ## conf args
    parser.add_argument('--est_conf.find_best', action='store_true', help='find the best model')
    parser.add_argument('--est_conf.load_final', action='store_true', help='load the final model')
    #parser.add_argument('--est_conf.model', type=str, default='c+w', help='model name')
    parser.add_argument('--est_conf.eps', type=float, default=0.01, help='epsilon')
    parser.add_argument('--est_conf.T_max', type=float, default=1.0, help='T max range')
    parser.add_argument('--est_conf.T_min', type=float, default=1e-6, help='T min range')
    parser.add_argument('--est_conf.T_step', type=float, default=0.01, help='T step size')


    args = parser.parse_args()
    args = to_tree_namespace(args)

    ## duplicate
    args.train.save_root = os.path.join(args.snapshot_root, args.exp_name)
    args.train_base.save_root = args.train.save_root
    args.train_iw.save_root = args.train.save_root
    args.cal_iw.save_root = args.train.save_root
    args.train_advtr.save_root = args.train.save_root
    args.est_conf.save_root = args.train.save_root
    
    args.model.n_labels = args.data.n_labels
    args.model.img_size = args.data.img_size

    args.train_advtr.schedule_reg_param_adv = not args.train_advtr.no_adv_reg_schedule

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
        
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)



