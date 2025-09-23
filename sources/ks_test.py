# Reff
# https://github.com/vinyluis/Articles/blob/main/Kolmogorov-Smirnov/Kolmogorov-Smirnov%20-%20Classification.ipynb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp
import scipy

def cdf(sample, x, sort = False):
    '''
    Return the value of the Cumulative Distribution Function, evaluated for a given sample and a value x.
    
    Args:
        sample: The list or array of observations.
        x: The value for which the numerical cdf is evaluated.
    
    Returns:
        cdf = CDF_{sample}(x)
    '''
    
    # Sorts the sample, if needed
    if sort:
        sample.sort()
    
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    
    return cdf

def ks_2samp(samp1, samp2):
    sample1 = np.array(samp1)
    sample2 = np.array(samp2)
    # Gets all observations
    observations = np.concatenate((sample1, sample2))
    observations.sort()
    # Sorts the samples
    sample1.sort()
    sample2.sort()
    # Evaluates the KS statistic
    D_ks = [] # KS Statistic list
    cdf1 = []
    cdf2 = []
    for x in observations:
        cdf_sample1 = cdf(sample = sample1, x  = x)
        cdf_sample2 = cdf(sample = sample2, x  = x)
        
        cdf1.append(cdf_sample1)
        cdf2.append(cdf_sample2)
        
        D_ks.append(abs(cdf_sample1 - cdf_sample2))
        
    ks_stat = max(D_ks)
    idx_max = np.where(D_ks == ks_stat)
    xks_stat = observations[idx_max]
    # Calculates the P-Value based on the two-sided test
    # The P-Value comes from the KS Distribution Survival Function (SF = 1-CDF)
    m, n = float(len(sample1)), float(len(sample2))
    en = m * n / (m + n)
    p_value = stats.kstwo.sf(ks_stat, np.round(en))
    ks_value = ks_stat
    return {"ks_stat": ks_stat, "p_value" : p_value}, xks_stat, observations, cdf1, cdf2, ks_value 


def plot_ks2sample_cdf(sample1, sample2, n_bins = 10, xlim = [], feature = '', colname=" ", irac = False):
    pl_name = ''
    rand_num = np.random.randint(20)
    if irac == True:
        pl_name = 'irac'+str(rand_num)
    else:
        pl_name = feature
    ks_plots = os.getcwd() + "/plots/ks/" #'/users/walter/git/mightee-cosmos-classification/mightee-class/plots/ks_test/'
    cwd = os.getcwd()
    os.chdir(ks_plots)
    
    ks_stats, xks_stat, obs, cdf1, cdf2, ks_value  = ks_2samp(sample1[feature], sample2[feature])
    rows = 5
    print( ks_stats)
    plt.figure(figsize=(10, 8))
    ax = plt.subplot2grid((3,3),(2,0),rowspan = 1, colspan = 2)
    ax_x = plt.subplot2grid((3,3),(0,0),rowspan = 2, colspan = 2, sharex=ax)
    
    # Plot the histograms
    ax_x.hist(np.array(sample2[feature]), bins = n_bins, 
                 histtype = "step", linewidth = 3, alpha= 1, color= "r", label = 'SFG')
    ax_x.hist(np.array(sample1[feature]), bins = n_bins, 
                 histtype = "step", linewidth = 3, alpha= 1, color= "b", label = 'AGN')
    
    agn_mean = np.mean(np.array(sample2[feature]))
    ax_x.axvline(x= agn_mean,lw= 2, linestyle = '--', alpha=0.5, color= "r")
    
    sfg_mean = np.mean(np.array(sample1[feature]))
    ax_x.axvline(x=sfg_mean,lw= 2, linestyle = '--', alpha=0.5, color= "b")

    # The Mean of two Populations
    print('The mean difference: ', np.abs(agn_mean - sfg_mean))
    
    ax_x.set_ylabel('Number (N)', fontsize =20)
    ax_x.tick_params(axis='y', which='major', labelsize=16)
    
    
    ax_x.legend(fontsize=16)
    ax_x.label_outer()
    
    # ax_x.set_xlim(xlim)
    # ax_x.tick_params(axis='both', which='major', labelsize=16)

    # Plot the ks_test
    ax.plot(obs, np.array(cdf2), label = 'SFG', lw= 3, c = 'r', alpha= 1)
    ax.plot(obs, np.array(cdf1), label = 'AGN', lw= 3, c = 'b', alpha= 1)
    ax.axvline(x=xks_stat[0], color='k',lw= 2, linestyle = '--', alpha=0.5)
    text = 'KS = '+str(round(ks_value,2))
    text1 = 'KS = '+"{:.2f}".format(ks_value)
    # print('Formatted Value: ',text1)
    ax.text(0.95, 0.05, text1, transform=ax.transAxes,
        fontsize=15, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    # ax.sharex(ax_x)
    ax.set_xlabel(colname, fontsize =20)
    ax.set_ylabel(r'$(\Sigma N)/N$', fontsize =20)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.set_xlim(xlim)

    plt.subplots_adjust(hspace=0)
    
    plt.setp(ax_x.get_xticklabels(), visible=False)
    # The y-ticks will overlap with "hspace=0", so we'll hide the bottom tick
    ax_x.set_yticks(ax_x.get_yticks()[1:]) 
    # ax.set_xticks(ax.get_xticks()[1:]) 

    # plt.tight_layout()
    # plt.rcParams['figure.figsize'] = [10, 8]
    # plt.rcParams['figure.dpi'] = 100
    plt.savefig(ks_plots+pl_name+'.pdf', bbox_inches='tight')
    plt.show()
    
    os.chdir(cwd)