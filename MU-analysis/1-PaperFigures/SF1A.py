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
import scipy.stats as sts

# Computes the analysis for Supplementary Figure 1A
# This is the analysis that compares 0.2 to 0.4 deg diameter stimuli
# TODO compare RF to 2RF


save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
mat_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/PSTHmats/'

MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_mn_data = pkl.load(f)

with open(S_dir + 'vari_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_vr_data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)


diams = np.array(list(diams_data.keys())).round(1)
del(diams_data)

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

def fit_mean_response(meanR,diam):
    if meanR.shape[0] != len(diam):
        diam = diam[1:]

    args = (diam,meanR)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x
    diams_tight = np.logspace(np.log10(diam[0]),np.log10(diam[-1]),1000)
    Rhat = dalib.ROG(diams_tight,*popt)
    RF_diam = diams_tight[np.argmax(Rhat)]
    nearSUR2_idx = np.argmin(np.abs(2*RF_diam - diam))
    nearSUR3_idx = np.argmin(np.abs(3*RF_diam - diam))
    RFdiam_idx   = np.argmin(np.abs(RF_diam - diam))
    
    return RFdiam_idx, nearSUR2_idx, nearSUR3_idx


eps = 0.0000001
# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

mn_SML = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_RF  = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_RFa = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_SR2 = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_SR3 = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_LAR = np.nan * np.ones((len(SG_mn_data), anal_duration))

vr_SML = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_RF  = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_RFa = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_SR2 = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_SR3 = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_LAR = np.nan * np.ones((len(SG_mn_data), anal_duration))

for i, u in enumerate(SG_mn_data.keys()):
    mn_matrix = SG_mn_data[u]
    vr_matrix = SG_vr_data[u]
    meanR = mn_matrix[:,first_tp:last_tp].mean(axis=1)
    RFdiam_idx, nearSUR2_idx, nearSUR3_idx = fit_mean_response(meanR,diams)

    if mn_matrix.shape[0] != 19:
        # nean responses 
        mn_SML[i,:] = mn_matrix[0,first_tp:last_tp]
        mn_RF[i,:]  = mn_matrix[1,first_tp:last_tp]
        mn_RFa[i,:] = mn_matrix[RFdiam_idx-1,first_tp:last_tp]
        mn_SR2[i,:] = mn_matrix[nearSUR2_idx-1,first_tp:last_tp]
        mn_SR3[i,:] = mn_matrix[nearSUR3_idx-1,first_tp:last_tp]
        mn_LAR[i,:] = mn_matrix[-1,first_tp:last_tp]

        # variances
        vr_SML[i,:] = vr_matrix[0,first_tp:last_tp]
        vr_RF[i,:]  = vr_matrix[1,first_tp:last_tp]
        vr_RFa[i,:] = vr_matrix[RFdiam_idx-1,first_tp:last_tp]
        vr_SR2[i,:] = vr_matrix[nearSUR2_idx-1,first_tp:last_tp]
        vr_SR3[i,:] = vr_matrix[nearSUR3_idx-1,first_tp:last_tp]
        vr_LAR[i,:] = vr_matrix[-1,first_tp:last_tp]

    else:
        mn_SML[i,:] = mn_matrix[1,first_tp:last_tp]
        mn_RF[i,:]  = mn_matrix[2,first_tp:last_tp]        
        mn_RFa[i,:] = mn_matrix[RFdiam_idx,first_tp:last_tp]
        mn_SR2[i,:] = mn_matrix[nearSUR2_idx,first_tp:last_tp]
        mn_SR3[i,:] = mn_matrix[nearSUR3_idx,first_tp:last_tp]
        mn_LAR[i,:] = mn_matrix[-1,first_tp:last_tp]

        vr_SML[i,:] = vr_matrix[1,first_tp:last_tp]
        vr_RF[i,:]  = vr_matrix[2,first_tp:last_tp]
        vr_RFa[i,:] = vr_matrix[RFdiam_idx,first_tp:last_tp]
        vr_SR2[i,:] = vr_matrix[nearSUR2_idx,first_tp:last_tp]
        vr_SR3[i,:] = vr_matrix[nearSUR3_idx,first_tp:last_tp]
        vr_LAR[i,:] = vr_matrix[-1,first_tp:last_tp]

count_bins = np.arange(0,30,1)
fano_SML, fano_RF, mean_SML, mean_RF, fano_boot_SML, fano_boot_RF,bmeans_SML, bmeans_RF = dalib.PSTH_meanmatch_twopopulations(mn_SML, mn_RF, vr_SML, vr_RF, count_bins,100)
fano_RF2, fano_LAR, mean_RF2, mean_LAR, fano_boot_RF2, fano_boot_LAR, bmeans_RF2, bmeans_LAR = dalib.PSTH_meanmatch_twopopulations(mn_RF, mn_LAR, vr_RF, vr_LAR, count_bins,100)

fano_RFa, fano_SR2, mean_RFa, mean_SR2, fano_boot_RFa, fano_boot_SR2,bmeans_RFa, bmeans_SR2 = dalib.PSTH_meanmatch_twopopulations(mn_RFa, mn_SR2, vr_RFa, vr_SR2, count_bins,100)
fano_RFa_new, fano_SR3, mean_RFa_new, mean_SR3, fano_boot_RFa_new, fano_boot_SR3,bmeans_RFa_new, bmeans_SR3 = dalib.PSTH_meanmatch_twopopulations(mn_RFa, mn_SR3, vr_RFa, vr_SR3, count_bins,100)

# RUNS UP TO HERE, NOW PLOTTING!

t1 = 50
t = np.linspace(t1,t1+anal_duration,anal_duration)

fig, axes = plt.subplots(4,2,sharex=True)
this_row = 0
this_col = 0
axes[this_row,this_col].plot(t,mean_SML/0.1,'-',color='grey',label='0.2')
axes[this_row,this_col].plot(t,mean_RF/0.1,'-',color='orange',label='0.4')
axes[this_row,this_col].set_ylim([0,100])
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 0
axes[this_row,this_col].fill_between(t,(mean_SML/0.1) - np.nanstd(bmeans_SML/0.1,axis=0),(mean_SML/0.1) + np.nanstd(bmeans_SML/0.1,axis=0),color='grey',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_RF/0.1) - np.nanstd(bmeans_RF/0.1,axis=0),(mean_RF/0.1) + np.nanstd(bmeans_RF/0.1,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].set_ylim([0,100])

this_row = 2
this_col = 0
axes[this_row,this_col].plot(t,fano_SML,'-',color='grey',label='0.2')
axes[this_row,this_col].plot(t,fano_RF,'-',color='orange',label='0.4')
axes[this_row,this_col].set_ylim([0,5])
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 0
axes[this_row,this_col].fill_between(t,(fano_SML) - np.nanstd(fano_boot_SML,axis=0),(fano_SML) + np.nanstd(fano_boot_SML,axis=0),color='grey',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_RF) - np.nanstd(fano_boot_RF,axis=0),(fano_RF) + np.nanstd(fano_boot_RF,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].set_ylim([0,5])

##
this_row = 0
this_col = 1
axes[this_row,this_col].plot(t,mean_RF2/0.1,'-',color='orange',label='0.2')
axes[this_row,this_col].plot(t,mean_LAR/0.1,'-',color='blue',label='0.4')
axes[this_row,this_col].set_ylim([0,60])
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 1
axes[this_row,this_col].fill_between(t,(mean_RF2/0.1) - np.nanstd(bmeans_RF2/0.1,axis=0),(mean_RF2/0.1) + np.nanstd(bmeans_RF2/0.1,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_LAR/0.1) - np.nanstd(bmeans_LAR/0.1,axis=0),(mean_LAR/0.1) + np.nanstd(bmeans_LAR/0.1,axis=0),color='blue',alpha=0.5)
axes[this_row,this_col].set_ylim([0,60])

this_row = 2
this_col = 1
axes[this_row,this_col].plot(t,fano_RF2,'-',color='orange',label='0.4')
axes[this_row,this_col].plot(t,fano_LAR,'-',color='blue',label='26')
axes[this_row,this_col].set_ylim([0,5])
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 1
axes[this_row,this_col].fill_between(t,(fano_RF2) - np.nanstd(fano_boot_RF2,axis=0),(fano_RF2) + np.nanstd(fano_boot_RF2,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_LAR) - np.nanstd(fano_boot_LAR,axis=0),(fano_LAR) + np.nanstd(fano_boot_LAR,axis=0),color='blue',alpha=0.5)
axes[this_row,this_col].set_ylim([0,5])

if save_figures:
    plt.savefig(fig_dir+'SG-mean-matched-PSTHs.svg')

plt.figure(2)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    [np.mean(mean_SML/0.1),np.mean(mean_RF/0.1)],
    yerr=[np.std(mean_SML/0.1),np.std(mean_RF/0.1)],
    color=['grey','orange'])
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])
ax.set_ylabel('Firing rate (Hz)')

ax = plt.subplot(2,2,2)
ax.bar([1,2],
    [np.mean(mean_RF2/0.1),np.mean(mean_LAR/0.1)],
    yerr=[np.std(mean_RF2/0.1),np.std(mean_LAR/0.1)],
    color=['orange','blue'])
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])
#ax.set_ylabel('Firing rate (Hz)')

ax = plt.subplot(2,2,3)
ax.bar([1,2],
    [np.mean(fano_SML),np.mean(fano_RF)],
    yerr=[np.std(fano_SML),np.std(fano_RF)],
    color=['grey','orange'])

ax.set_ylabel('Fano factor')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])

ax = plt.subplot(2,2,4)
ax.bar([1,2],
    [np.mean(fano_RF2),np.mean(fano_LAR)],
    yerr=[np.std(fano_RF2),np.std(fano_LAR)],
    color=['orange','blue'])

ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])

if save_figures:
    plt.savefig(fig_dir+'SG-mean-matched-averages.svg')


print('Fano factor for 0.2: ' + str(np.mean(fano_SML))+ ' +/- ' + str(np.std(fano_SML)))
print('Fano factor for 0.4: ' + str(np.mean(fano_RF))+ ' +/- ' + str(np.std(fano_RF)))
print('p :', sts.ttest_ind(fano_SML,fano_RF))
print('####################')
print('Fano factor for 0.4: ' + str(np.mean(fano_RF2))+ ' +/- ' + str(np.std(fano_RF2)))
print('Fano factor for 26: ' + str(np.mean(fano_LAR))+ ' +/- ' + str(np.std(fano_LAR)))
print('p :', sts.ttest_ind(fano_RF2,fano_LAR))
