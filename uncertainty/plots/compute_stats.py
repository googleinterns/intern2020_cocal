import os, sys
import argparse
import types
import glob
import numpy as np
import pickle

import tensorflow as tf

import data
import model 
from util import *
from learning import LearnerCls, LearnerDACls, LearnerClsSelf, LearnerConfPred
from learning import TempScalingCls as CalibratorCls

##TODO: clean-up tf options
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)


def main(args):

    # ## init a snapshot path
    # os.makedirs(args.train.save_root, exist_ok=True)

    # ## init logger
    # sys.stdout = Logger(os.path.join(args.train.save_root, 'out'))

    # ## print args
    # print_args(args)


    snap_list = glob.glob(args.snapshot_prefix + '_*')
    print(snap_list)
    print("# experiments = ", len(snap_list))

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
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[0],
        resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False, ##TODO: check if it's necessary
    )
    assert(len(args.aug_params) == 1) ##TODO
    ds_tar = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        aug_list=args.aug_params[0],
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[1])
    ds_dom = data.DomainDataset(
        data.MultiSourceDataset(
            args.data.src,
            args.aug_params,
            batch_size=args.data.batch_size,
            train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True, test_aug=True, # augment all splits
            domain_id=1,
            color=False if args.data.img_size[2]==1 else True,
            size=args.data.img_size[0],
            sample_ratio=args.data.sample_ratio[0],
            resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False, ##TODO: check if it's necessary
        ),
        getattr(data, args.data.tar)(
            root=os.path.join('data', args.data.tar.lower()),
            batch_size=args.data.batch_size,
            aug_list=args.aug_params[0],
            train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True, test_aug=True, # augment all splits
            domain_id=0,
            color=False if args.data.img_size[2]==1 else True,
            size=args.data.img_size[0],
            sample_ratio=args.data.sample_ratio[1]))

    ##TODO: redundant
    ds_src_init = data.MultiSourceDataset(
        args.data.src,
        args.aug_params_init,
        batch_size=args.data.batch_size,
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],##TODO
        sample_ratio=args.data.sample_ratio[0],
        resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False, ##TODO: check if it's necessary
    )
    assert(len(args.aug_params) == 1) ##TODO
    ds_tar_init = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        aug_list=args.aug_params_init[0],##TODO
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[1])
    ds_dom_init = data.DomainDataset(
        data.MultiSourceDataset(
            args.data.src,
            args.aug_params_init,
            batch_size=args.data.batch_size,
            train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True, test_aug=True, # augment all splits
            domain_id=1,
            color=False if args.data.img_size[2]==1 else True,
            size=args.data.img_size[0],
            sample_ratio=args.data.sample_ratio[0],
            resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False, ##TODO: check if it's necessary
        ),
        getattr(data, args.data.tar)(
            root=os.path.join('data', args.data.tar.lower()),
            batch_size=args.data.batch_size,
            aug_list=args.aug_params_init[0], ##TODO
            train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True, test_aug=True, # augment all splits
            domain_id=0,
            color=False if args.data.img_size[2]==1 else True,
            size=args.data.img_size[0],
            sample_ratio=args.data.sample_ratio[1]))

    ##TODO: redundant
    ds_src_self = data.MultiSourceDataset(
        args.data.src,
        args.aug_params,
        batch_size=args.data.batch_size,
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        domain_id=1,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[0],
        resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False, ##TODO: check if...
    )
    assert(len(args.aug_params) == 1) ##TODO
    ds_tar_self = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        aug_list=args.aug_params[0],
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        domain_id=0,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[1],
        double_aug=True if args.training_type=='selfcon' else False,
    )
    print()    

    if args.merge_train_val:
        ds_src.train = data.ChainLoader(ds_src.train, ds_src.val)
        ds_dom.train = data.ChainLoader(ds_dom.train, ds_dom.val)

    ## collect stats
    cls_error_init_list, cal_error_init_list = [], []
    cls_error_list, cal_error_list = [], []
    perf_epoch_list = []
    for snap_root in snap_list:

        ##
        ## final student
        ##
        
        ## a student model
        mdl_st_base = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
        mdl_st = model.Student(args.model, mdl_st_base, ds_src_self, ds_tar_self, ideal=args.ideal)

        ## load the final student
        mdl_st.model_base.load_weights(os.path.join(snap_root, 'model_params_final'))

        ## evaluate
        learner = LearnerClsSelf(None, None,  mdl_st, None)
        error, ece, *_ = learner.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        
        cls_error_list = np.append(cls_error_list, error.numpy())
        cal_error_list = np.append(cal_error_list, ece)

        print(f"[final, {args.snapshot_prefix}, cls error, n = {len(cls_error_list)}] mean = {np.mean(cls_error_list*100.0):.2f}%, std = {np.std(cls_error_list*100.0):.2f}%")
        print(f"[final, {args.snapshot_prefix}, cal error, n = {len(cal_error_list)}] mean = {np.mean(cal_error_list*100.0):.2f}%, std = {np.std(cal_error_list*100.0):.2f}%")

        ##
        ## init student
        ##
        
        ## load the init student
        mdl_fn_init = os.path.basename(glob.glob(os.path.join(snap_root, 'model_params_*init*.index'))[0])
        mdl_fn_init = mdl_fn_init[:mdl_fn_init.rfind('_')]
        if 'sourceonly' in mdl_fn_init:
            mdl_st.model_base.load_weights(os.path.join(snap_root, mdl_fn_init+'_best'))
            learner = LearnerClsSelf(None, None,  mdl_st, None)

        else:
            assert('advtr' in mdl_fn_init)
            ## init a adv model
            mdl_adv = getattr(model, args.train_advtr.model_advtr)(n_in=mdl_st.model_base.dim_feat)
            mdl_st_adv = model.DAN(mdl_st.model_base, mdl_adv)
            mdl_st_adv.load_weights(os.path.join(snap_root, mdl_fn_init+'_final'))

            ## init a learner
            learner = LearnerDACls(None, mdl_st_adv)

            
        ## evaluate
        error, ece, *_ = learner.test(ds_tar.test, ld_name=args.data.tar, verbose=True)

        cls_error_init_list = np.append(cls_error_init_list, error.numpy())
        cal_error_init_list = np.append(cal_error_init_list, ece)

        print(f"[init, {args.snapshot_prefix}, cls error, n = {len(cls_error_init_list)}] mean = {np.mean(cls_error_init_list*100.0):.2f}%, std = {np.std(cls_error_init_list*100.0):.2f}%")
        print(f"[init, {args.snapshot_prefix}, cal error, n = {len(cal_error_init_list)}] mean = {np.mean(cal_error_init_list*100.0):.2f}%, std = {np.std(cal_error_init_list*100.0):.2f}%")

        ##
        ## teacher performance at each step
        ##
        if args.no_mid_results:
            continue
        
        cls_error_epoch_list, cal_error_epoch_list, prec_epoch_list, cov_epoch_list = [], [], [], []
        for i_epoch in range(1, args.train.n_epochs): # ignore the last
            ## load
            print("!!!! currently load best, but may load final later")
            mdl_st.model_base.load_weights(os.path.join(snap_root, f'model_params_base_epoch_{i_epoch}_best'))
            
            ## cls/cal error
            learner = LearnerClsSelf(None, None,  mdl_st, None)
            error, ece, *_ = learner.test(ds_tar.test, ld_name=args.data.tar, verbose=True)

            print(error.numpy(), ece)
           
            ## precision/coverage
            learner = LearnerConfPred(None, mdl_st.model_conf, mdl_st.model_base)
            ## set a constant
            mdl_st.model_conf.T = tf.Variable(1.0 - args.train_conf.eps) 
            ## test the model
            prec, n_conf, n = learner.test(ds_tar.test, ld_name='tar', verbose=True)

            cls_error_epoch_list.append(error.numpy())
            cal_error_epoch_list.append(ece)
            prec_epoch_list.append(prec.numpy())
            cov_epoch_list.append(float(n_conf.numpy())/float(n))

        perf_epoch_list.append({
            'cls_error': np.array(cls_error_epoch_list),
            'cal_error': np.array(cal_error_epoch_list),
            'prec': np.array(prec_epoch_list),
            'cov': np.array(cov_epoch_list)})

        print()


    ## save
    fn = args.snapshot_prefix + '.pk'
    pickle.dump(
        {
            'cls_error_init': cls_error_init_list,
            'cal_error_init': cal_error_init_list,
            'cls_error': cls_error_list,
            'cal_error': cal_error_list,
            'perf_epoch': perf_epoch_list
        },
        open(fn, 'wb'))
        



def init_aug_params(aug, args):
    aug_params = []
    for a in aug:
        if a == 'jitter':
            aug_params.append([('jitter', {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4})])
            
        # elif a == 'shake':
        #     args.aug_params.append([('randaug', {'size': 32, 'mode': 'SHAKE'})])
            
        elif a == 'svhnspec':
            aug_params.append([ 
                ('intensity_flip', {}),
                ('intensity_scaling', {'min': -1.5, 'max': 1.5}),
                ('intensity_offset', {'min': -0.5, 'max': 0.5}),
                ('affine', {'std': 0.1}),
                ('translation', {'x_max': 2.0, 'y_max': 2.0}),
                ('gaussian', {'std': 0.1}),
            ])
        elif a == 'translation':
            aug_params.append([ 
                ('translation', {'x_max': 2.0, 'y_max': 2.0}),
            ])
        elif a == 'randaug':
            aug_params.append([('randaug', {'size': args.data.img_size[0]})])            
        else:
            ##TODO: simplify
            aug_params.append(None)
    return aug_params

            
def parse_args():
    ## inint a parser
    parser = argparse.ArgumentParser(description='digit dataset training')

    ## meta args
    parser.add_argument('--snapshot_prefix', type=str, required=True)
    parser.add_argument('--no_mid_results', action='store_true')
    
    #parser.add_argument('--exp_name', required=True, type=str, help='experiment name')
    #parser.add_argument('--snapshot_root', default='snapshots', type=str, help='snapshot root name')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--ideal', action='store_true', help='enable cheatkey')
    parser.add_argument('--merge_train_val', action='store_true', help='merge train and validataion set')
    parser.add_argument('--training_type', type=str, default='selfcon', help='snapshot root name') ## selfcon, self, advtr, srconly

    ## dataset args
    parser.add_argument('--data.batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--data.n_labels', default=10, type=int, help='the number of labels')
    parser.add_argument('--data.src', type=str, nargs='*', default=['MNIST'], help='list of sources')
    parser.add_argument('--data.tar', type=str, default='USPS', help='target') 
    parser.add_argument('--data.aug', type=str, nargs='*', default=[''], help='list of data augmentation')
    parser.add_argument('--data.aug_init', type=str, nargs='*', default=[''], help='list of data augmentation')
    parser.add_argument('--data.img_size', type=int, nargs=3, default=(32, 32, 3), help='image size')
    parser.add_argument('--data.sample_ratio', type=float, nargs=2, default=[1.0, 1.0])
    
    ## model args
    parser.add_argument('--model.base', default='ResNet18', type=str, help='model name')
    parser.add_argument('--model.conf', default='ConfPred', type=str, help='model name')
    parser.add_argument('--model.iw', default='BigFNN', type=str, help='model name')

    # ## self-train args
    # parser.add_argument('--train.rerun', action='store_true', help='find the best model')
    # #parser.add_argument('--train.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train.n_epochs', type=int, default=50, help='the number of training iterations')
    # parser.add_argument('--train.init_advtr', action='store_true', help='model initialization approach')
    # parser.add_argument('--train.val_period', default=1, type=int, help='validation period in epochs')
    
    # ## base model train args
    # parser.add_argument('--train_base.rerun', action='store_true', help='find the best model')
    # #parser.add_argument('--train_base.load_final', action='store_true', help='load the final model')
    # parser.add_argument('--train_base.optim', default='SGD', type=str, help='optimizer')
    # parser.add_argument('--train_base.lr', default=0.01, type=float, help='learning rate')
    # parser.add_argument('--train_base.lr_step_size', default=5, type=float, help='stepsize for step learning rate scheduler')
    # parser.add_argument('--train_base.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    # parser.add_argument('--train_base.weight_decay', type=float, default=0.0, help='L2 weight decay')
    # parser.add_argument('--train_base.momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--train_base.n_epochs', default=25, type=int, help='the number of epochs')
    # parser.add_argument('--train_base.val_period', default=1, type=int, help='validation period in epochs')

    # ## iw train args
    # parser.add_argument('--train_iw.rerun', action='store_true', help='find the best model')
    # parser.add_argument('--train_iw.load_final', action='store_true', help='load the final model')    
    # parser.add_argument('--train_iw.optim', default='SGD', type=str, help='optimizer')
    # parser.add_argument('--train_iw.lr', default=0.01, type=float, help='learning rate')
    # parser.add_argument('--train_iw.lr_step_size', default=20, type=float, help='stepsize for step learning rate scheduler')
    # parser.add_argument('--train_iw.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    # parser.add_argument('--train_iw.weight_decay', type=float, default=0.0, help='L2 weight decay')
    # parser.add_argument('--train_iw.momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--train_iw.n_epochs', default=100, type=int, help='the number of epochs')
    # parser.add_argument('--train_iw.val_period', default=1, type=int, help='validation period in epochs')

    # ## cal args
    # parser.add_argument('--cal_iw.rerun', action='store_true', help='find the best model')
    # parser.add_argument('--cal_iw.load_final', action='store_true', help='load the final model')    
    # parser.add_argument('--cal_iw.optim', default='SGD', type=str, help='optimizer')
    # parser.add_argument('--cal_iw.lr', default=0.01, type=float, help='learning rate')
    # parser.add_argument('--cal_iw.lr_step_size', default=50, type=float, help='stepsize for step learning rate scheduler')
    # parser.add_argument('--cal_iw.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    # parser.add_argument('--cal_iw.weight_decay', type=float, default=0.0, help='L2 weight decay')
    # parser.add_argument('--cal_iw.momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--cal_iw.n_epochs', default=500, type=int, help='the number of epochs')
    # parser.add_argument('--cal_iw.val_period', default=1, type=int, help='validation period in epochs')

    # ## train args
    # parser.add_argument('--train_advtr.rerun', action='store_true', help='find the best model')
    # #parser.add_argument('--train_advtr.load_final', action='store_true', help='load the final model')
    # parser.add_argument('--train_advtr.optim', default='SGD', type=str, help='optimizer')
    # parser.add_argument('--train_advtr.lr', default=0.01, type=float, help='learning rate')
    # parser.add_argument('--train_advtr.lr_step_size', default=20, type=float, help='stepsize for step learning rate scheduler')
    # parser.add_argument('--train_advtr.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    # parser.add_argument('--train_advtr.weight_decay', type=float, default=0.0, help='L2 weight decay')
    # parser.add_argument('--train_advtr.momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--train_advtr.n_epochs', default=100, type=int, help='the number of epochs')
    # parser.add_argument('--train_advtr.val_period', default=1, type=int, help='validation period in epochs')
    # parser.add_argument('--train_advtr.advtr_type', type=str, default='DANN', help='domain-adversarial training type')

    parser.add_argument('--train_advtr.model_advtr', type=str, default='BigAdvFNN', help='adversarial network name')
    # parser.add_argument('--train_advtr.reg_param_adv', type=float, default=1.0, help='adversarial loss regularization parameter')
    # parser.add_argument('--train_advtr.no_adv_reg_schedule', action='store_true', help='do not schedule the adversarial loss regularization parameter')

    # ## base model init train args
    # parser.add_argument('--train_base_init.rerun', action='store_true', help='find the best model')
    # parser.add_argument('--train_base_init.load_final', action='store_true', help='load the final model')
    # parser.add_argument('--train_base_init.optim', default='SGD', type=str, help='optimizer')
    # parser.add_argument('--train_base_init.lr', default=0.01, type=float, help='learning rate')
    # parser.add_argument('--train_base_init.lr_step_size', default=20, type=float, help='stepsize for step learning rate scheduler')
    # parser.add_argument('--train_base_init.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    # parser.add_argument('--train_base_init.weight_decay', type=float, default=0.0, help='L2 weight decay')
    # parser.add_argument('--train_base_init.momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--train_base_init.n_epochs', default=100, type=int, help='the number of epochs')
    # parser.add_argument('--train_base_init.val_period', default=1, type=int, help='validation period in epochs')

    # ## conf args
    # #parser.add_argument('--train_conf.rerun', action='store_true', help='find the best model')
    # #parser.add_argument('--train_conf.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train_conf.eps', type=float, default=0.01, help='epsilon')



    args = parser.parse_args()
    args = to_tree_namespace(args)

    ## duplicate
    ##TODO: better way?
    # args.train.save_root = os.path.join(args.snapshot_root, args.exp_name)
    # args.train_base.save_root = args.train.save_root
    # args.train_iw.save_root = args.train.save_root
    # args.cal_iw.save_root = args.train.save_root
    # args.train_advtr.save_root = args.train.save_root
    # args.train_base_init.save_root = args.train.save_root
    # args.train_conf.save_root = args.train.save_root
    
    args.model.n_labels = args.data.n_labels
    args.model.img_size = args.data.img_size

    # args.train_advtr.schedule_reg_param_adv = not args.train_advtr.no_adv_reg_schedule
    # args.train_advtr.load_final = True

    #args.train.load_final = True
    #args.train_base.load_final = True

    

    ## init aug parameters
    args.aug_params = init_aug_params(args.data.aug, args)
    args.aug_params_init = init_aug_params(args.data.aug_init, args)
    
    # args.aug_params = []
    # for a in args.data.aug:
    #     if a == 'jitter':
    #         args.aug_params.append([('jitter', {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4})])
            
    #     # elif a == 'shake':
    #     #     args.aug_params.append([('randaug', {'size': 32, 'mode': 'SHAKE'})])
            
    #     elif a == 'svhnspec':
    #         args.aug_params.append([ 
    #             ('intensity_flip', {}),
    #             ('intensity_scaling', {'min': -1.5, 'max': 1.5}),
    #             ('intensity_offset', {'min': -0.5, 'max': 0.5}),
    #             ('affine', {'std': 0.1}),
    #             ('translation', {'x_max': 2.0, 'y_max': 2.0}),
    #             ('gaussian', {'std': 0.1}),
    #         ])
    #     elif a == 'translation':
    #         args.aug_params.append([ 
    #             ('translation', {'x_max': 2.0, 'y_max': 2.0}),
    #         ])
    #     elif a == 'randaug':
    #         args.aug_params.append([('randaug', {'size': args.data.img_size[0]})])            
    #     else:
    #         ##TODO: simplify
    #         args.aug_params.append(None)
        
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)



