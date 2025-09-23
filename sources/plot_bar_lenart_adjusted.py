import os
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib as mpl
from uncertainties import unumpy as unp
from astropy.table import Table
import seaborn as sns
from rich import print
#from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from sources.config import *


def plot_classification_bar(df):
    
    # print(df)

    radio_loud_AGN = df['RLAGN']
    radio_loud_not_AGN = df['RQ']

    xray_AGN = df['XAGN']
    xray_not_AGN = df['notXAGN']

    mid_infra_AGN = df['midIRAGN']
    mid_infra_not_AGN = df['notmidIRAGN']

    opt_AGN = df['optAGN']
    opt_not_AGN = df['notoptAGN']

    vlba_AGN = df['VLBAAGN']
    
    AGN = df['AGN']
    not_AGN = df['SFG']
    maybe_AGN = df['probSFG']
    not_classified = len(df['AGN']) - (len(df[df['SFG'] == True]) + len(df[df['AGN'] == True])  + len(df[df['probSFG'] == True]))
    print('not classified', not_classified)


    fig, ax1 = plt.subplots(figsize=(10,8))
    # title = d.par.plot_title+".classification_barchart"

    ax1.minorticks_on()
    # remove minor ticks on x axis
    ax1.xaxis.set_tick_params(which='minor', bottom=False)



    # the scatter plot:
    # sigma = d.par.sigma_err
    #ax1.bar('all sources', len(df), color=colorC1)
    bars = []


    ax1_AGN = ax1.bar(r'$\bf{Total}$', sum(AGN), color=colorB1, edgecolor=colorC1)
    # ax1_maybe_not_AGN = ax1.bar(r'$\bf{Total}$', sum(maybe_not_AGN), bottom=sum(AGN)+sum(maybe_AGN), color=colorA1+[0.5], edgecolor=colorC1)
    ax1_not_AGN = ax1.bar(r'$\bf{Total}$', sum(not_AGN), bottom=sum(AGN), color=colorA1, edgecolor=colorC1)
    ax1_maybe_AGN = ax1.bar(r'$\bf{Total}$', sum(maybe_AGN), bottom=sum(AGN)+ sum(not_AGN), color=colorA1+[0.5], edgecolor=colorC1)
    
    ax1_not_classified = ax1.bar(r'$\bf{Total}$', not_classified, bottom = sum(AGN)+sum(maybe_AGN)+sum(not_AGN), color=silver, alpha=0.5, edgecolor=colorC1)
    bars += [ax1_AGN, ax1_not_AGN]
    count_AGN = sum(AGN)
    count_maybe_AGN = sum(maybe_AGN)
    # count_maybe_not_AGN = sum(maybe_not_AGN)
    count_not_AGN = sum(not_AGN)
    count_not_classified = not_classified

    perc_AGN = round(count_AGN/len(df)*100, 1)
    perc_maybe_AGN = round(count_maybe_AGN/len(df)*100, 1)
    # perc_maybe_not_AGN = round(count_maybe_not_AGN/len(df)*100, 1)
    perc_not_AGN = round(count_not_AGN/len(df)*100, 1)
    perc_not_classified = round(count_not_classified/len(df)*100, 1)

    #ax1.bar('not_classified', sum(df['not_classified']), color="gray")

    ax1.bar('Radio-loud', len(df), color=silver, alpha=0.5)
    ax1.bar('Radio-loud', sum(radio_loud_AGN), color=colorB1)
    ax1_RQ = ax1_not_XAGN = ax1.bar('Radio-loud', sum(radio_loud_not_AGN), bottom=sum(radio_loud_AGN), color=colorB1+[0.5])
    
    ax1.bar('Mid-infrared', len(df), color=silver, alpha=0.5)
    ax1.bar('Mid-infrared', sum(mid_infra_AGN), color=colorB1)
    ax1_not_midAGN = ax1.bar('Mid-infrared', sum(mid_infra_not_AGN), bottom=sum(mid_infra_AGN), color=y)#colorA1)


    ax1.bar('Optical', len(df), color=silver, alpha=0.5)
    ax1.bar('Optical', sum(opt_AGN), color=colorB1)
    ax1.bar('Optical', sum(opt_not_AGN), bottom=sum(opt_AGN), color=y)#colorA1)

    
    ax1.bar('X-ray', len(df), color=silver, alpha=0.5)
    ax1.bar('X-ray', sum(xray_AGN), color=colorB1)
    ax1.bar('X-ray', sum(xray_not_AGN), bottom=sum(xray_AGN), color=y)#colorA1)

    ax1.bar('VLBI', len(df), color=silver, alpha=0.5)
    ax1.bar('VLBI', sum(vlba_AGN), color=colorB1)





    #plt.yscale("log")
    #plt.xscale("log")
    #plt.xlim([1e8, 1e12])
    plt.ylim([0, len(df)])

    #ax1.set_xlabel(r'classification type', fontsize=15)
    ax1.set_ylabel(r'Source counts', fontsize=20)
    plt.xticks(rotation = 20, fontsize=16)
    ax1.set_yticklabels(['0', '1k', '2k', '3k', '4k', '5k'], fontsize=16)

    ax2 = ax1.twinx()
    ax2.minorticks_on()
    ax2.set_ylabel(r'', fontsize=20)
    ax2.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax2.set_yticklabels(["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], fontsize=16)
    ax2.set_yticks([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], minor=True)
    ax2.grid( which='major', linestyle='dashed')
    #ax2.grid(b=True, which='minor', linestyle='dotted')



    #ax1.xaxis.set_major_formatter(ScalarFormatter())
    
    #ax1_tick_labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #ax1.set_xticks(ax1_tick_labels)
    #ax1.set_xticklabels([x-1 for x in ax1_tick_labels])

    blank = ax1.hlines([0], 0, 0, color=colorA1, alpha=0.0, linewidth=0.0)

    ax2.legend((ax1_AGN, ax1_not_AGN, ax1_maybe_AGN, ax1_not_classified,ax1_RQ, ax1_not_midAGN  ),
            (
            f"AGN ({count_AGN}, {perc_AGN}%)",
            f"SFGs ({count_not_AGN}, {perc_not_AGN}%)",
            f"probSFGs ({count_maybe_AGN}, {perc_maybe_AGN}%)",
            f"unclassified ({count_not_classified}, {perc_not_classified}%)",
            f"RQ",
            f"non-AGN"),
            frameon=True, fancybox=True, markerscale=2, facecolor='white', prop={"size":13}
            )


    #plt.legend(frameon=True, fancybox=True, markerscale=2)
    plt.tight_layout()

    # outfile_full_path = os.path.join(DIR_OUTPUT, d.par.title, title)
    # print(f"Saving file: {outfile_full_path}.pdf and .png")
    plt.savefig('criteria.pdf')
    # plt.savefig(outfile_full_path+".png")
    # plt.close()
    plt.show()