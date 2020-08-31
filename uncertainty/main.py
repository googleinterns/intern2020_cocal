import os, sys
import argparse
import types

import tensorflow as tf

import data
import model 
from util import *
from learning import LearnerCls, LearnerDACls, LearnerClsSelf
from learning import TempScalingCls as CalibratorCls

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(args):

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
    ds_src, ds_tar, ds_dom, ds_src_init, ds_tar_init, ds_dom_init, ds_src_self, ds_tar_self = init_datasets(args)
    print()
    
    ## train
    if args.training_type == 'selfcon' or args.training_type == 'self':

        if args.training_type == 'selfcon':
            print('## train approach: self-training + consistency')
        else:
            print('## train approach: self-training')

        ####
        ## student teacher learning
        ####
        mdl_st_base = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
        mdl_st = model.Student(args.model, mdl_st_base, ds_src_self, ds_tar_self, ideal=args.ideal)

        mdl_tc_base = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
        mdl_tc = model.Teacher(args.model, mdl_tc_base, ds_src_self, ds_tar_self, ideal=args.ideal)

        ## init a learner
        learner = LearnerClsSelf(args.train, args.train_base, mdl_st, mdl_tc,
                                 params_conf=args.train_conf,
                                 params_init=args.train_advtr if args.train.init_advtr else args.train_base_init)
        ## train the model
        learner.train(ds_src, ds_tar, ds_dom, ds_src_init, ds_tar_init, ds_dom_init)
        ## test the model
        learner.test(ds_tar.test, ld_name=args.data.tar, verbose=True)
        learner.test(ds_src.test, ld_name='src', verbose=True)
        print()

        
    elif args.training_type == 'srconly':
        print('## train approach: source-only')

        mdl = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
        ## init a learner
        learner = LearnerCls(args.train_base_init, mdl)
        ## train the model
        learner.train(ds_src.train, ds_src.val, ds_tar.test)
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        print()
        
    elif args.training_type == 'taronly':
        print('## train approach: target-only')

        mdl = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
        ## init a learner
        learner = LearnerCls(args.train_base_init, mdl)
        ## train the model
        learner.train(ds_tar.train, ds_tar.val, ds_tar.test)
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        print()

    elif args.training_type == 'advtr':
        mdl = getattr(model, args.model.base)(num_class=args.model.n_labels, input_shape=args.model.img_size)
        mdl_adv = getattr(model, args.train_advtr.model_advtr)(n_in=mdl.dim_feat)
        
        ## init a learner
        learner = LearnerDACls(args.train_advtr, model.DAN(mdl, mdl_adv))
        ## train the model
        learner.train([ds_src.train, ds_dom.train], None, ds_tar.test)        
        ## test the model
        learner.test(ds_tar.test, ld_name='tar', verbose=True)
        print()

        

def init_aug_params(aug, args):
    aug_params = []
    for a in aug:
        if a == 'jitter':
            aug_params.append([('jitter', {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4})])            
        elif a == 'svhnspec' or a == 'svhnaug':
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
            aug_params.append(None)
    return aug_params


def init_datasets(args):
    ds_src = data.MultiSourceDataset(
        args.data.src,
        args.aug_params,
        batch_size=args.data.batch_size,
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[0],
        resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False,
    )
    assert(len(args.aug_params) == 1)
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
            resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False,
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

    ds_src_init = data.MultiSourceDataset(
        args.data.src,
        args.aug_params_init,
        batch_size=args.data.batch_size,
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[0],
        resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False,
    )
    assert(len(args.aug_params) == 1)
    ds_tar_init = getattr(data, args.data.tar)(
        root=os.path.join('data', args.data.tar.lower()),
        batch_size=args.data.batch_size,
        aug_list=args.aug_params_init[0],
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
            resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False,
        ),
        getattr(data, args.data.tar)(
            root=os.path.join('data', args.data.tar.lower()),
            batch_size=args.data.batch_size,
            aug_list=args.aug_params_init[0],
            train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True, test_aug=True, # augment all splits
            domain_id=0,
            color=False if args.data.img_size[2]==1 else True,
            size=args.data.img_size[0],
            sample_ratio=args.data.sample_ratio[1]))

    ds_src_self = data.MultiSourceDataset(
        args.data.src,
        args.aug_params,
        batch_size=args.data.batch_size,
        train_shuffle=True, train_aug=True, val_shuffle=True, val_aug=True,
        domain_id=1,
        color=False if args.data.img_size[2]==1 else True,
        size=args.data.img_size[0],
        sample_ratio=args.data.sample_ratio[0],
        resize_pad=True if len(args.data.src)==1 and args.data.src[0]=='MNIST' and args.data.tar=='SVHN' else False,
    )
    assert(len(args.aug_params) == 1)
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

    return ds_src, ds_tar, ds_dom, ds_src_init, ds_tar_init, ds_dom_init, ds_src_self, ds_tar_self


def parse_args():
    ## inint a parser
    parser = argparse.ArgumentParser(description='digit dataset training')

    ## meta args
    parser.add_argument('--exp_name', required=True, type=str, help='experiment name')
    parser.add_argument('--snapshot_root', default='snapshots', type=str, help='snapshot root name')
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

    ## self-train args
    parser.add_argument('--train.rerun', action='store_true', help='find the best model')
    #parser.add_argument('--train.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train.n_epochs', type=int, default=50, help='the number of training iterations')
    parser.add_argument('--train.init_advtr', action='store_true', help='model initialization approach')
    parser.add_argument('--train.val_period', default=1, type=int, help='validation period in epochs')
    
    ## base model train args
    parser.add_argument('--train_base.rerun', action='store_true', help='find the best model')
    parser.add_argument('--train_base.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train_base.optim', default='SGD', type=str, help='optimizer')
    parser.add_argument('--train_base.lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--train_base.lr_step_size', default=5, type=float, help='stepsize for step learning rate scheduler')
    parser.add_argument('--train_base.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    parser.add_argument('--train_base.weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('--train_base.momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--train_base.n_epochs', default=25, type=int, help='the number of epochs')
    parser.add_argument('--train_base.val_period', default=1, type=int, help='validation period in epochs')

    ## iw train args
    parser.add_argument('--train_iw.rerun', action='store_true', help='find the best model')
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
    parser.add_argument('--cal_iw.rerun', action='store_true', help='find the best model')
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
    parser.add_argument('--train_advtr.rerun', action='store_true', help='find the best model')
    #parser.add_argument('--train_advtr.load_final', action='store_true', help='load the final model')
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
    parser.add_argument('--train_advtr.no_adv_reg_schedule', action='store_true', help='do not schedule the adversarial loss regularization parameter')

    ## base model init train args
    parser.add_argument('--train_base_init.rerun', action='store_true', help='find the best model')
    parser.add_argument('--train_base_init.load_final', action='store_true', help='load the final model')
    parser.add_argument('--train_base_init.optim', default='SGD', type=str, help='optimizer')
    parser.add_argument('--train_base_init.lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--train_base_init.lr_step_size', default=20, type=float, help='stepsize for step learning rate scheduler')
    parser.add_argument('--train_base_init.lr_step_decay_rate', default=0.5, type=float, help='decay rate for step learning rate scheduler')
    parser.add_argument('--train_base_init.weight_decay', type=float, default=0.0, help='L2 weight decay')
    parser.add_argument('--train_base_init.momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--train_base_init.n_epochs', default=100, type=int, help='the number of epochs')
    parser.add_argument('--train_base_init.val_period', default=1, type=int, help='validation period in epochs')

    ## conf args
    parser.add_argument('--train_conf.eps', type=float, default=0.01, help='epsilon')
    
    args = parser.parse_args()
    args = to_tree_namespace(args)

    ## duplicate
    args.train.save_root = os.path.join(args.snapshot_root, args.exp_name)
    args.train_base.save_root = args.train.save_root
    args.train_iw.save_root = args.train.save_root
    args.cal_iw.save_root = args.train.save_root
    args.train_advtr.save_root = args.train.save_root
    args.train_base_init.save_root = args.train.save_root
    args.train_conf.save_root = args.train.save_root
    
    args.model.n_labels = args.data.n_labels
    args.model.img_size = args.data.img_size

    args.train_advtr.schedule_reg_param_adv = not args.train_advtr.no_adv_reg_schedule

    # always load the final
    args.train_advtr.load_final = True
    args.train.load_final = True

    ## init aug parameters
    args.aug_params = init_aug_params(args.data.aug, args)
    args.aug_params_init = init_aug_params(args.data.aug_init, args)
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)



