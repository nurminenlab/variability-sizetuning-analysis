import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats

#import pdb

save_figures = False

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 250
bsl_end   = bsl_begin + anal_duration

eps = 0.0000001

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)

diams = np.array(list(diams_data.keys()))
del(diams_data)

with open(S_dir + 'mean_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_mn_data = pkl.load(f)    
with open(S_dir + 'vari_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_vr_data = pkl.load(f)    


count_window = 100
# containers for mean and fano PSTHs
SG_mean = np.ones((len(SG_mn_data),19,1000))*np.nan
SG_fano = np.ones((len(SG_mn_data),19,1000))*np.nan
G_mean = np.ones((len(G_mn_data),19,1000))*np.nan
G_fano = np.ones((len(G_mn_data),19,1000))*np.nan
IG_mean = np.ones((len(IG_mn_data),19,1000))*np.nan
IG_fano = np.ones((len(IG_mn_data),19,1000))*np.nan

# collect SG data
for unit_indx, unit in enumerate(list(SG_mn_data.keys())):
    # loop diams
    mn_mtrx = SG_mn_data[unit]
    vr_mtrx = SG_vr_data[unit]
    
    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus             
            SG_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            SG_fano[unit_indx,stim+1,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])
        else:           
            SG_mean[unit_indx,stim,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            SG_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])



plt.figure(1,figsize=(1.335, 1.115))
# plot smallest stimulus
ax  = plt.subplot(211)
ax2 = ax.twinx()
t = np.arange(-150,600,1)
stim = 0
fano_PSTH    = np.nanmean(SG_fano[:,stim,bsl_begin:],axis=0)
fano_PSTH_SE = np.nanstd(SG_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(SG_fano.shape[0])

PSTH    = np.nanmean(SG_mean[:,stim,bsl_begin:],axis=0)
PSTH_SE = np.nanstd(SG_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(SG_fano.shape[0])

ax.fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red')
ax.plot(t,fano_PSTH,'-',color=[0.5, 0, 0])
ax.plot([t[0], t[-1]], [1, 1], 'r--')
ax.set_ylim([0.5, 2.0])
ax.set_xlim([-150,600])
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red',labelsize=8)
ax.yaxis.label.set_color('red')
ax.spines['top'].set_visible(False)

ax2.fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray')
ax2.plot(t,PSTH,'k-')
ax2.plot([t[0], t[-1]], [1, 1], 'k--')
ax2.set_ylim([0, 20])
ax2.set_xlim([-150,600])
ax2.spines['left'].set_color('red')
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='y',labelsize=8)

# 1 deg
ax  = plt.subplot(212)
ax2 = ax.twinx()
t = np.arange(-150,600,1)
stim = 5
fano_PSTH    = np.nanmean(SG_fano[:,stim,bsl_begin:],axis=0)
fano_PSTH_SE = np.nanstd(SG_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(SG_fano.shape[0])

PSTH    = np.nanmean(SG_mean[:,stim,bsl_begin:],axis=0)
PSTH_SE = np.nanstd(SG_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(SG_fano.shape[0])

ax.fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red')
ax.plot(t,fano_PSTH,'-',color=[0.5, 0, 0])
ax.plot([t[0], t[-1]], [1, 1], 'r--')
ax.set_ylim([0.5, 2.0])
ax.set_xlim([-150,600])
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red',labelsize=8)
ax.yaxis.label.set_color('red')
ax.spines['top'].set_visible(False)

ax2.fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray')
ax2.plot(t,PSTH,'k-')
ax2.plot([t[0], t[-1]], [1, 1], 'k--')
ax2.set_ylim([0, 20])
ax2.set_xlim([-150,600])
ax2.spines['left'].set_color('red')
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='y',labelsize=8)

if save_figures:
    plt.savefig(fig_dir + 'F3B_PSTH_SG_mean_normalized-fano_plot.svg')

# collect G data
#------------------------------------------------------------------------------
for unit_indx, unit in enumerate(list(G_mn_data.keys())):
    # loop diams
    mn_mtrx = G_mn_data[unit]
    vr_mtrx = G_vr_data[unit]
    
    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus 
            G_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            G_fano[unit_indx,stim+1,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])
        else:
            G_mean[unit_indx,stim,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            G_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])



plt.figure(2,figsize=(1.335, 1.115))
# plot smallest stimulus
ax  = plt.subplot(211)
ax2 = ax.twinx()
t = np.arange(-150,600,1)
stim = 0
fano_PSTH    = np.nanmean(G_fano[:,stim,bsl_begin:],axis=0)
fano_PSTH_SE = np.nanstd(G_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(G_fano.shape[0])

PSTH    = np.nanmean(G_mean[:,stim,bsl_begin:],axis=0)
PSTH_SE = np.nanstd(G_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(G_fano.shape[0])

ax.fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red')
ax.plot(t,fano_PSTH,'-',color=[0.5, 0, 0])
ax.plot([t[0], t[-1]], [1, 1], 'r--')
ax.set_ylim([0.5, 2.0])
ax.set_xlim([-150,600])
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red',labelsize=8)
ax.yaxis.label.set_color('red')
ax.spines['top'].set_visible(False)

ax2.fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray')
ax2.plot(t,PSTH,'k-')
ax2.plot([t[0], t[-1]], [1, 1], 'k--')
ax2.set_ylim([0, 20])
ax2.set_xlim([-150,600])
ax2.spines['left'].set_color('red')
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='y',labelsize=8)

# 1 deg
ax  = plt.subplot(212)
ax2 = ax.twinx()
t = np.arange(-150,600,1)
stim = 5
fano_PSTH    = np.nanmean(G_fano[:,stim,bsl_begin:],axis=0)
fano_PSTH_SE = np.nanstd(G_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(G_fano.shape[0])

PSTH    = np.nanmean(G_mean[:,stim,bsl_begin:],axis=0)
PSTH_SE = np.nanstd(G_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(G_fano.shape[0])

ax.fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red')
ax.plot(t,fano_PSTH,'-',color=[0.5, 0, 0])
ax.plot([t[0], t[-1]], [1, 1], 'r--')
ax.set_ylim([0.5, 2.0])
ax.set_xlim([-150,600])
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red',labelsize=8)
ax.yaxis.label.set_color('red')
ax.spines['top'].set_visible(False)

ax2.fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray')
ax2.plot(t,PSTH,'k-')
ax2.plot([t[0], t[-1]], [1, 1], 'k--')
ax2.set_ylim([0, 20])
ax2.set_xlim([-150,600])
ax2.spines['left'].set_color('red')
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='y',labelsize=8)

if save_figures:
    plt.savefig(fig_dir + 'F3B_PSTH_G_mean_normalized-fano_plot.svg')

# collect IG data
#------------------------------------------------------------------------------
for unit_indx, unit in enumerate(list(IG_mn_data.keys())):
    # loop diams
    mn_mtrx = IG_mn_data[unit]
    vr_mtrx = IG_vr_data[unit]
    
    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus 
            IG_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            IG_fano[unit_indx,stim+1,:] = FF/np.mean(FF[bsl_begin:bsl_begin + 151])
        else:
            IG_mean[unit_indx,stim,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            IG_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])

plt.figure(3,figsize=(1.335, 1.115))
# plot smallest stimulus
ax  = plt.subplot(211)
ax2 = ax.twinx()
t = np.arange(-150,600,1)
stim = 0
fano_PSTH    = np.nanmean(IG_fano[:,stim,bsl_begin:],axis=0)
fano_PSTH_SE = np.nanstd(IG_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(IG_fano.shape[0])

PSTH    = np.nanmean(IG_mean[:,stim,bsl_begin:],axis=0)
PSTH_SE = np.nanstd(IG_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(IG_fano.shape[0])

ax.fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red')
ax.plot(t,fano_PSTH,'-',color=[0.5, 0, 0])
ax.plot([t[0], t[-1]], [1, 1], 'r--')
ax.set_ylim([0.6, 1.3])
ax.set_xlim([-150,600])
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red',labelsize=8)
ax.yaxis.label.set_color('red')
ax.spines['top'].set_visible(False)

ax2.fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray')
ax2.plot(t,PSTH,'k-')
ax2.plot([t[0], t[-1]], [1, 1], 'k--')
ax2.set_ylim([0, 6])
ax2.set_xlim([-150,600])
ax2.spines['left'].set_color('red')
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='y',labelsize=8)

# 1 deg
ax  = plt.subplot(212)
ax2 = ax.twinx()
t = np.arange(-150,600,1)
stim = 5
fano_PSTH    = np.nanmean(IG_fano[:,stim,bsl_begin:],axis=0)
fano_PSTH_SE = np.nanstd(IG_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(IG_fano.shape[0])

PSTH    = np.nanmean(IG_mean[:,stim,bsl_begin:],axis=0)
PSTH_SE = np.nanstd(IG_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(IG_fano.shape[0])

ax.fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red')
ax.plot(t,fano_PSTH,'-',color=[0.5, 0, 0])
ax.plot([t[0], t[-1]], [1, 1], 'r--')
ax.set_ylim([0.6, 1.3])
ax.set_xlim([-150,600])
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red',labelsize=8)
ax.yaxis.label.set_color('red')
ax.spines['top'].set_visible(False)

ax2.fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray')
ax2.plot(t,PSTH,'k-')
ax2.plot([t[0], t[-1]], [1, 1], 'k--')
ax2.set_ylim([0, 6])
ax2.set_xlim([-150,600])
ax2.spines['left'].set_color('red')
ax2.spines['top'].set_visible(False)
ax2.tick_params(axis='y',labelsize=8)

if save_figures:
    plt.savefig(fig_dir + 'F3B_PSTH_IG_mean_normalized-fano_plot.svg')

# compute stats
SG_fano_means = np.ones((SG_fano.shape[0],))
for i in range(SG_fano.shape[0]):
    if ~np.isnan(SG_fano[i,0,0]):
        SG_fano_means[i] = np.mean(SG_fano[i,0,first_tp:last_tp])
    else:
        SG_fano_means[i] = np.mean(SG_fano[i,1,first_tp:last_tp])

G_fano_means = np.ones((G_fano.shape[0],))
for i in range(G_fano.shape[0]):
    if ~np.isnan(G_fano[i,0,0]):
        G_fano_means[i] = np.mean(G_fano[i,0,first_tp:last_tp])
    else:
        G_fano_means[i] = np.mean(G_fano[i,1,first_tp:last_tp])

IG_fano_means = np.ones((IG_fano.shape[0],))
for i in range(IG_fano.shape[0]):
    if ~np.isnan(IG_fano[i,0,0]):
        IG_fano_means[i] = np.mean(IG_fano[i,1,first_tp:last_tp])
    else:
        IG_fano_means[i] = np.mean(IG_fano[i,1,first_tp:last_tp])


print('SG')
print(stats.ttest_1samp(SG_fano_means,1,nan_policy='omit',alternative='greater'))
print('G')
print(stats.ttest_1samp(G_fano_means,1,nan_policy='omit',alternative='greater'))
print('IG')
print(stats.ttest_1samp(IG_fano_means,1,nan_policy='omit',alternative='greater'))