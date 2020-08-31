import os, sys
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_rel_diag(n_bins, conf_t, conf_e, n_cnt, ece, fn, fontsize=15):
    bins = np.linspace(0.0, 1.0, n_bins)
    bin_center = (bins[:-1] + bins[1:])/2.0
    conf_e, conf_t = conf_e[n_cnt>0], conf_t[n_cnt>0] 
    plt.figure(1)
    plt.clf()
    fig, ax1 = plt.subplots()
    ## acc-conf plot
    h1 = ax1.plot(conf_e, conf_t, 'ro--', label='estimated')
    h2 = ax1.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k-', label='ideal')
    ## example rate
    ax2 = ax1.twinx()
    h3 = ax2.bar(bin_center, n_cnt/np.sum(n_cnt), width=(bin_center[1]-bin_center[0])*0.75, color='b', edgecolor='k', alpha=0.5, label='ratio')
    ## beautify
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax2.set_xlim((0, 1))
    ax2.set_ylim((0, 1))
    ax1.grid('on')
    ax1.set_xlabel('confidence', fontsize=fontsize)
    ax1.set_ylabel('accuracy', fontsize=fontsize)
    ax2.set_ylabel('example ratio', fontsize=fontsize)
    plt.title('ECE = %.2f%%'%(ece*100.0), fontsize=fontsize)
    plt.legend(handles=[h1[0], h2[0], h3], loc='upper left', fontsize=fontsize)
    fig.tight_layout()
    ## save
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()


def plot_acc_rank(corr, log_conf, fn, fontsize=15, ratio=0.01):

    ## sort
    corr = corr[np.argsort(log_conf, kind='stable')][::-1]  # conduct a stable sorting to properly handle tie
    
    n = len(corr)

    ranking = [float(i) for i in range(1, n+1)]
    corr_mean = [corr[:i].mean() for i in range(1, n+1)]

    n_trim = round(n*ratio)
    ranking = ranking[:n_trim]
    corr_mean = corr_mean[:n_trim]

    ## plot
    plt.figure(1)
    plt.clf()
    plt.plot(ranking, corr_mean, 'r--')

    # beautify
    plt.grid('on')
    plt.ylim((0.0, 1.0))
    plt.xlabel('ranking', fontsize=fontsize)
    plt.ylabel('average accuracy', fontsize=fontsize)
    
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()

    
def plot_acc_conf(corr, conf, fn, fontsize=15):

    conf_rng = np.arange(0.0, 1.0, 0.01)
    corr_mean = np.array([corr[conf>=c].mean() for c in conf_rng])
    n_cnt = np.array([np.sum(conf>=c) for c in conf_rng])

    ## plot
    plt.figure(1)
    plt.clf()
    fig, ax1 = plt.subplots()

    ## #example 
    ax2 = ax1.twinx()
    bin_center = conf_rng
    h2 = ax2.bar(bin_center, n_cnt, width=(bin_center[1]-bin_center[0]), color='b', edgecolor=None, alpha=0.3, label='#examples')

    ## curve
    h1 = ax1.plot(conf_rng, corr_mean, 'r--', label='conditional accuracy')

    # beautify
    ax1.set_xlim((0, 1))
    ax1.set_ylim((0, 1))
    ax2.set_xlim((0, 1))

    ax1.grid('on')
    ax1.set_xlabel('confidence threshold', fontsize=fontsize)
    ax1.set_ylabel('conditional accuracy', fontsize=fontsize)
    ax2.set_ylabel('#examples', fontsize=fontsize)
    plt.legend(handles=[h2, h1[0]], fontsize=fontsize, loc='lower left')
    
    plt.savefig(fn+'.png', bbox_inches='tight')
    plt.close()
    
    
def ECE(ph, yh, y, n_bins=15, overconf=False, rel_diag_fn=None):
    assert(len(ph) == len(y))
    n = len(y)
    bins = np.linspace(0.0, 1.0, n_bins)
    conf_e = np.zeros(len(bins)-1)
    conf_t = np.zeros(len(bins)-1)
    n_cnt = np.zeros(len(bins)-1)
    
    for i, (l, u) in enumerate(zip(bins[:-1], bins[1:])):
        idx = (ph>=l)&(ph<=u) if i==(n_bins-2) else (ph>=l)&(ph<u)
        if np.sum(idx) == 0:
            continue
        ph_i, yh_i, y_i = ph[idx], yh[idx], y[idx]
        ## compute (estimated) true confidence
        conf_t[i] = np.mean((yh_i == y_i).astype(np.float32))
        ## compute estimated confidence
        conf_e[i] = np.mean(ph_i)
        ## count the examples in the bin
        n_cnt[i] = np.sum(idx).astype(np.float32)
        
    ## expected calibration error
    ece = np.sum(np.abs(conf_e - conf_t)*n_cnt/n)
    if overconf:
        ece_oc = np.sum(np.maximum(0.0, conf_e - conf_t)*n_cnt/n)
        
    ## plot a reliability diagram
    if rel_diag_fn is not None:
        plot_rel_diag(n_bins, conf_t, conf_e, n_cnt, ece, rel_diag_fn)

    if overconf:
        return ece, ece_oc
    else:
        return ece
    


## https://gist.github.com/DavidWalz/8538435
def bino_ci(k, n, alpha=1e-5):
    lo = stats.beta.ppf(alpha/2, k, n-k+1)
    hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi
