import os, sys
import argparse
import types
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# import tensorflow as tf

# import data
# import model 
# from util import *
# from learning import LearnerCls, LearnerDACls, LearnerClsRT, LearnerConfPred
# from learning import TempScalingCls as CalibratorCls

# ##TODO: clean-up tf options
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

def plot_prec_cov(prec, cov, labels, fn, fontsize=15):

    plt.figure(1)
    plt.clf()

    h1 = plt.plot(cov[0], prec[0], 'rs-', label=labels[0])
    h2 = plt.plot(cov[1], prec[1], 'bs-', label=labels[1])

    plt.xlabel('coverage (%)', fontsize=fontsize)
    plt.ylabel('precision (%)', fontsize=fontsize)
    plt.grid('on')
    plt.legend(handles=[h1[0], h2[0]])
    plt.savefig(fn+'.png', bbox_inches='tight')


def main():

    fig_fn = 'snapshots_final_presentation/prec_cov'

    data_fn = 'snapshots_final_presentation/summary_m2s_selfcon_svhnaug_advtr_svhnaug.pk'
    data = pickle.load(open(data_fn, 'rb'))

    prec_advtr = np.array(data['perf_epoch'][0]['prec']) * 100.0
    cov_advtr = np.array(data['perf_epoch'][0]['cov'] ) * 100.0

    data_fn = 'snapshots_final_presentation/summary_m2s_selfcon_svhnaug_srconly_svhnaug.pk'
    data = pickle.load(open(data_fn, 'rb'))

    prec_srconly = np.array(data['perf_epoch'][0]['prec']) * 100.0
    cov_srconly = np.array(data['perf_epoch'][0]['cov'] ) * 100.0

    plot_prec_cov([prec_advtr, prec_srconly], [cov_advtr, cov_srconly], ['advtr+svhn-aug init', 'srconly+svhn-aug init'], fig_fn)



if __name__ == '__main__':
    main()



