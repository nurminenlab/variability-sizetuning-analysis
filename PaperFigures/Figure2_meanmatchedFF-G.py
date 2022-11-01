import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import basinhopping, curve_fit
import scipy.io as scio

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
mat_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/PSTHmats/'

MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    SG_mn_data = pkl.load(f)

with open(S_dir + 'vari_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    SG_vr_data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_G-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)


diams = np.array(list(diams_data.keys())).round(1)
del(diams_data)

eps = 0.0000001
# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

mn_SML = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_RF  = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_LAR = np.nan * np.ones((len(SG_mn_data), anal_duration))

vr_SML = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_RF  = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_LAR = np.nan * np.ones((len(SG_mn_data), anal_duration))

for i, u in enumerate(SG_mn_data.keys()):
    mn_matrix = SG_mn_data[u]
    vr_matrix = SG_vr_data[u]
    if mn_matrix.shape[0] != 19:
        mn_SML[i,:] = mn_matrix[0,first_tp:last_tp]
        mn_RF[i,:]  = mn_matrix[1,first_tp:last_tp]
        mn_LAR[i,:] = mn_matrix[-1,first_tp:last_tp]
        vr_SML[i,:] = vr_matrix[0,first_tp:last_tp]
        vr_RF[i,:]  = vr_matrix[1,first_tp:last_tp]
    else:
        mn_SML[i,:] = mn_matrix[1,first_tp:last_tp]
        mn_RF[i,:]  = mn_matrix[2,first_tp:last_tp]
        mn_LAR[i,:] = mn_matrix[-1,first_tp:last_tp]
        vr_SML[i,:] = vr_matrix[1,first_tp:last_tp]
        vr_RF[i,:]  = vr_matrix[2,first_tp:last_tp]
        vr_LAR[i,:] = vr_matrix[-1,first_tp:last_tp]

count_bins = np.arange(0,30,1)
fano_SML, fano_RF, mean_SML, mean_RF, fano_boot_SML, fano_boot_RF,bmeans_SML, bmeans_RF = dalib.PSTH_meanmatch_twopopulations(mn_SML, mn_RF, vr_SML, vr_RF, count_bins,100)
fano_RF2, fano_LAR, mean_RF2, mean_LAR, fano_boot_RF2, fano_boot_LAR, bmeans_RF2, bmeans_LAR = dalib.PSTH_meanmatch_twopopulations(mn_RF, mn_LAR, vr_RF, vr_LAR, count_bins,100)

t1 = 50
t = np.linspace(t1,t1+anal_duration,anal_duration)

fig, axes = plt.subplots(4,2,sharex=True)
this_row = 0
this_col = 0
axes[this_row,this_col].plot(t,mean_SML/0.1,'b-',label='0.2')
axes[this_row,this_col].plot(t,mean_RF/0.1,'r-',label='0.4')
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 0
axes[this_row,this_col].fill_between(t,(mean_SML/0.1) - np.nanstd(bmeans_SML/0.1,axis=0),(mean_SML/0.1) + np.nanstd(bmeans_SML/0.1,axis=0),color='b',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_RF/0.1) - np.nanstd(bmeans_RF/0.1,axis=0),(mean_RF/0.1) + np.nanstd(bmeans_RF/0.1,axis=0),color='r',alpha=0.5)

this_row = 2
this_col = 0
axes[this_row,this_col].plot(t,fano_SML,'b-',label='0.2')
axes[this_row,this_col].plot(t,fano_RF,'r-',label='0.4')
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 0
axes[this_row,this_col].fill_between(t,(fano_SML) - np.nanstd(fano_boot_SML,axis=0),(fano_SML) + np.nanstd(fano_boot_SML,axis=0),color='b',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_RF) - np.nanstd(fano_boot_RF,axis=0),(fano_RF) + np.nanstd(fano_boot_RF,axis=0),color='r',alpha=0.5)

##
this_row = 0
this_col = 1
axes[this_row,this_col].plot(t,mean_RF2/0.1,'b-',label='0.2')
axes[this_row,this_col].plot(t,mean_LAR/0.1,'r-',label='0.4')
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 1
axes[this_row,this_col].fill_between(t,(mean_RF2/0.1) - np.nanstd(bmeans_RF2/0.1,axis=0),(mean_RF2/0.1) + np.nanstd(bmeans_RF2/0.1,axis=0),color='b',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_LAR/0.1) - np.nanstd(bmeans_LAR/0.1,axis=0),(mean_LAR/0.1) + np.nanstd(bmeans_LAR/0.1,axis=0),color='r',alpha=0.5)

this_row = 2
this_col = 1
axes[this_row,this_col].plot(t,fano_RF2,'b-',label='0.2')
axes[this_row,this_col].plot(t,fano_LAR,'r-',label='0.4')
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 1
axes[this_row,this_col].fill_between(t,(fano_RF2) - np.nanstd(fano_boot_RF2,axis=0),(fano_RF2) + np.nanstd(fano_boot_RF2,axis=0),color='b',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_LAR) - np.nanstd(fano_boot_LAR,axis=0),(fano_LAR) + np.nanstd(fano_boot_LAR,axis=0),color='r',alpha=0.5)

if save_figures:
    plt.savefig(fig_dir+'F2-G-mean-matched-PSTHs.svg')

plt.figure(2)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(fano_SML),np.mean(fano_RF)]),
    yerr=[3.96*np.nanstd(np.nanmean(fano_boot_SML,axis=1)),3.96*np.nanstd(np.nanmean(fano_boot_RF,axis=1))],
    color=['b','r'])

ax.set_ylabel('Fano factor')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])

ax = plt.subplot(2,2,3)
ax.bar([1,2],
    np.array([np.mean(mean_SML),
    np.mean(mean_RF)])/0.1,
    yerr=[np.std(np.mean(bmeans_SML/0.1,axis=1)),np.std(np.mean(bmeans_RF/0.1,axis=1))],
    color=['b','r'])
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])
ax.set_ylabel('Firing rate (Hz)')

if save_figures:
    plt.savefig(fig_dir+'F2-G-mean-matched-averages-SMLRF.svg')


plt.figure(3)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(fano_RF2),
    np.nanmean(fano_LAR)]),
    yerr=[3.96*np.nanstd(np.nanmean(fano_boot_RF2,axis=1)),3.96*np.nanstd(np.nanmean(fano_boot_LAR,axis=1))],
    color=['b','r'])
ax.set_ylabel('Fano factor')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])

ax = plt.subplot(2,2,3)
ax.bar([1,2],
    np.array([np.mean(mean_RF2),
    np.mean(mean_LAR)])/0.1,
    yerr=[np.std(np.nanmean(bmeans_RF2/0.1,axis=1)),np.std(np.nanmean(bmeans_LAR/0.1,axis=1))],
    color=['b','r'])
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])
ax.set_ylabel('Firing rate (Hz)')

if save_figures:
    plt.savefig(fig_dir+'F2-G-mean-matched-averages-RFLAR.svg')

