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

import pdb

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
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
            SG_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:]
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            SG_fano[unit_indx,stim+1,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])
        else:
            SG_mean[unit_indx,stim,:] = mn_mtrx[stim,:]
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            SG_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])

plt.figure()
row = 0
column = 0
fig, ax = plt.subplots(2,1)
ax_good = ax.ravel()
fig, ax2 = plt.subplots(2,1)
ax_fano = ax2.ravel()

t = np.arange(-150,600,1)
for plt_idx, stim in enumerate([0,5]):
    
    PSTH = np.nanmean((1000/count_window)*SG_mean[:,stim,bsl_begin:],axis=0)
    PSTH_SE = np.nanstd((1000/count_window)*SG_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(SG_mean.shape[0])

    fano_PSTH = np.nanmean(SG_fano[:,stim,bsl_begin:],axis=0)
    fano_PSTH_SE = np.nanstd(SG_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(SG_fano.shape[0])

    ax_good[plt_idx].fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray',alpha=0.5)
    ax_good[plt_idx].plot(t,PSTH,'k-')
    ax_good[plt_idx].set_xlim([-150,600])
    ax_good[plt_idx].set_ylim([0,150])

    ax_fano[plt_idx].fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red',alpha=0.5)
    ax_fano[plt_idx].plot(t,fano_PSTH,'r-')
    ax_fano[plt_idx].plot([t[0], t[-1]], [1, 1], 'k--')
    ax_fano[plt_idx].set_xlim([-150,600])


# collect G data
#------------------------------------------------------------------------------
for unit_indx, unit in enumerate(list(G_mn_data.keys())):
    # loop diams
    mn_mtrx = G_mn_data[unit]
    vr_mtrx = G_vr_data[unit]
    
    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus 
            G_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:]
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            G_fano[unit_indx,stim+1,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])
        else:
            G_mean[unit_indx,stim,:] = mn_mtrx[stim,:]
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            G_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])

plt.figure()
row = 0
column = 0
fig, ax = plt.subplots(2,1)
ax_good = ax.ravel()
fig, ax2 = plt.subplots(2,1)
ax_fano = ax2.ravel()

for plt_idx, stim in enumerate([0,5]):
    
    PSTH = np.nanmean((1000/count_window)*G_mean[:,stim,bsl_begin:],axis=0)
    PSTH_SE = np.nanstd((1000/count_window)*G_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(G_mean.shape[0])

    fano_PSTH = np.nanmean(G_fano[:,stim,bsl_begin:],axis=0)
    fano_PSTH_SE = np.nanstd(G_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(G_fano.shape[0])

    ax_good[plt_idx].fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray',alpha=0.5)
    ax_good[plt_idx].plot(t,PSTH,'k-')
    ax_good[plt_idx].set_xlim([-400,600])
    ax_good[plt_idx].set_ylim([0,150])

    ax_fano[plt_idx].fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red',alpha=0.5)
    ax_fano[plt_idx].plot(t,fano_PSTH,'r-')
    ax_fano[plt_idx].plot([t[0], t[-1]], [1, 1], 'k--')
    ax_fano[plt_idx].set_xlim([-150,600])

# collect IG data
#------------------------------------------------------------------------------
for unit_indx, unit in enumerate(list(IG_mn_data.keys())):
    # loop diams
    mn_mtrx = IG_mn_data[unit]
    vr_mtrx = IG_vr_data[unit]
    
    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus 
            IG_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:]
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            IG_fano[unit_indx,stim+1,:] = FF/np.mean(FF[bsl_begin:bsl_begin + 151])
        else:
            IG_mean[unit_indx,stim,:] = mn_mtrx[stim,:]
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            IG_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])

plt.figure()
fig, ax = plt.subplots(2,1)
ax_good = ax.ravel()
fig, ax2 = plt.subplots(2,1)
ax_fano = ax2.ravel()

for plt_idx, stim in enumerate([0,5]):
    
    PSTH = np.nanmean((1000/count_window)*IG_mean[:,stim,bsl_begin:],axis=0)
    PSTH_SE = np.nanstd((1000/count_window)*IG_mean[:,stim,bsl_begin:],axis=0)/np.sqrt(IG_mean.shape[0])

    fano_PSTH = np.nanmean(IG_fano[:,stim,bsl_begin:],axis=0)
    fano_PSTH_SE = np.nanstd(IG_fano[:,stim,bsl_begin:],axis=0)/np.sqrt(IG_fano.shape[0])

    ax_good[plt_idx].fill_between(t,PSTH-PSTH_SE,PSTH+PSTH_SE,color='gray',alpha=0.5)
    ax_good[plt_idx].plot(t,PSTH,'k-')
    ax_good[plt_idx].set_xlim([-400,600])
    ax_good[plt_idx].set_ylim([0,150])

    ax_fano[plt_idx].fill_between(t,fano_PSTH-fano_PSTH_SE,fano_PSTH+fano_PSTH_SE,color='red',alpha=0.5)
    ax_fano[plt_idx].plot(t,fano_PSTH,'r-')
    ax_fano[plt_idx].plot([t[0], t[-1]], [1, 1], 'k--')
    ax_fano[plt_idx].set_xlim([-150,600])


