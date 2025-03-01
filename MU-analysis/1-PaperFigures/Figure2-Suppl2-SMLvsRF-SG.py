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

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
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


def bootstrapped_p_for_sampled(sample1, sample2, mean1, mean2, n, n_boots = 1000):

    combined_samples = np.hstack((sample1, sample2))
    mean1_resampled  = np.random.choice(combined_samples, (n, n_boots), replace=True)
    mean2_resampled  = np.random.choice(combined_samples, (n, n_boots), replace=True)

    diffs = np.mean(mean1_resampled, axis=0) - np.mean(mean2_resampled, axis=0)
    if mean1 > mean2:
        p_value = np.sum(diffs > (mean1 - mean2)) / n_boots
    else:
        p_value = np.sum(diffs < (mean1 - mean2)) / n_boots

    return p_value

eps = 0.0000001
# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

mn_SML = np.nan * np.ones((len(SG_mn_data), anal_duration))
mn_RF  = np.nan * np.ones((len(SG_mn_data), anal_duration))

vr_SML = np.nan * np.ones((len(SG_mn_data), anal_duration))
vr_RF  = np.nan * np.ones((len(SG_mn_data), anal_duration))

for i, u in enumerate(SG_mn_data.keys()):
    mn_matrix = SG_mn_data[u]
    vr_matrix = SG_vr_data[u]
    meanR = mn_matrix[:,first_tp:last_tp].mean(axis=1)
    RFdiam_idx, nearSUR2_idx, nearSUR3_idx = fit_mean_response(meanR,diams)

    if mn_matrix.shape[0] != 19:
        # mean responses 
        mn_SML[i,:] = mn_matrix[0,first_tp:last_tp]
        mn_RF[i,:]  = mn_matrix[1,first_tp:last_tp]
        
        # variances
        vr_SML[i,:] = vr_matrix[0,first_tp:last_tp]
        vr_RF[i,:]  = vr_matrix[1,first_tp:last_tp]

    else:
        mn_SML[i,:] = mn_matrix[1,first_tp:last_tp]
        mn_RF[i,:]  = mn_matrix[2,first_tp:last_tp]        

        vr_SML[i,:] = vr_matrix[1,first_tp:last_tp]
        vr_RF[i,:]  = vr_matrix[2,first_tp:last_tp]


count_bins = np.arange(0,30,1)

fano_SML, fano_RF, mean_SML, mean_RF, fano_boot_SML, fano_boot_RF,bmeans_SML, bmeans_RF, N_SML, N_RF = dalib.PSTH_meanmatch_twopopulations(mn_SML, 
                                                                                                                                  mn_RF, 
                                                                                                                                  vr_SML, 
                                                                                                                                  vr_RF, 
                                                                                                                                  count_bins,
                                                                                                                                  1000)


# RUNS UP TO HERE, NOW PLOTTING!
t1 = 50
t = np.linspace(t1,t1+anal_duration,anal_duration)

fig, axes = plt.subplots(4,3,sharex=True)

# SML vs RF
#--------------------------
this_row = 0
this_col = 0
axes[this_row,this_col].plot(t,mean_SML/0.1,'-',color='grey',label='RF')
axes[this_row,this_col].plot(t,mean_RF/0.1,'-',color='orange',label='2RF')
axes[this_row,this_col].set_ylim([0,135])
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 0
axes[this_row,this_col].fill_between(t,(mean_SML/0.1) - np.nanstd(bmeans_SML/0.1,axis=0),(mean_SML/0.1) + np.nanstd(bmeans_SML/0.1,axis=0),color='grey',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_RF/0.1) - np.nanstd(bmeans_RF/0.1,axis=0),(mean_RF/0.1) + np.nanstd(bmeans_RF/0.1,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].plot(t,mean_SML/0.1,'-',color='grey') # mean response 
axes[this_row,this_col].plot(t,mean_RF/0.1,'-',color='orange') # mean response
axes[this_row,this_col].set_ylim([0,135])

this_row = 2
this_col = 0
axes[this_row,this_col].plot(t,fano_SML,'-',color='grey',label='RF')
axes[this_row,this_col].plot(t,fano_RF,'-',color='orange',label='2RF')
axes[this_row,this_col].set_ylim([0,5])
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 0
axes[this_row,this_col].fill_between(t,(fano_SML) - np.nanstd(fano_boot_SML,axis=0),(fano_SML) + np.nanstd(fano_boot_SML,axis=0),color='grey',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_RF) - np.nanstd(fano_boot_RF,axis=0),(fano_RF) + np.nanstd(fano_boot_RF,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].plot(t,fano_SML,'-',color='grey',label='RF')
axes[this_row,this_col].plot(t,fano_RF,'-',color='orange',label='2RF')
axes[this_row,this_col].set_ylim([0,5])

if save_figures:
    plt.savefig(fig_dir+'Figure2-Suppl-2-SG-mean-matched-PSTHs-SMLvsRF.svg')

plt.figure(2)
ax = plt.subplot(2,2,1)

SE_mean_boot_SML = np.nanmean(bmeans_SML,axis=1)
SE_mean_boot_RF = np.nanmean(bmeans_RF,axis=1)
# Compute the confidence interval
alpha = 0.001
SML_lower_bound = np.percentile(SE_mean_boot_SML, 100 * alpha / 2)
SML_upper_bound = np.percentile(SE_mean_boot_SML, 100 * (1 - alpha / 2))
RF_lower_bound = np.percentile(SE_mean_boot_RF, 100 * alpha / 2)
RF_upper_bound = np.percentile(SE_mean_boot_SML, 100 * (1 - alpha / 2))
# Compute the error bars
SML_error = np.array([np.mean(SE_mean_boot_SML) - SML_lower_bound, SML_upper_bound - np.mean(SE_mean_boot_SML)])
RF_error  = np.array([np.mean(SE_mean_boot_RF) - RF_lower_bound, RF_upper_bound - np.mean(SE_mean_boot_RF)])

ax.bar([1,2],
    [np.mean(SE_mean_boot_SML/0.1),np.mean(SE_mean_boot_RF/0.1)],
    yerr=[SML_error/0.1,RF_error/0.1],
    color=['grey','orange'])
ax.set_xticks([1,2])
ax.set_xticklabels(['0.1/0.2','RF'])
ax.set_ylabel('Firing rate (Hz)')

ax = plt.subplot(2,2,3)

SE_fano_boot_SML = np.nanmean(fano_boot_SML,axis=1)
SE_fano_boot_RF = np.nanmean(fano_boot_RF,axis=1)
# Compute the confidence interval
alpha = 0.001
SML_lower_bound = np.percentile(SE_fano_boot_SML, 100 * alpha / 2)
SML_upper_bound = np.percentile(SE_fano_boot_SML, 100 * (1 - alpha / 2))
RF_lower_bound = np.percentile(SE_fano_boot_RF, 100 * alpha / 2)
RF_upper_bound = np.percentile(SE_fano_boot_RF, 100 * (1 - alpha / 2))
# Compute the error bars
SML_error = [np.mean(SE_fano_boot_SML) - SML_lower_bound, SML_upper_bound - np.mean(SE_fano_boot_SML)]
RF_error = [np.mean(SE_fano_boot_RF) - RF_lower_bound, RF_upper_bound - np.mean(SE_fano_boot_RF)]

ax.bar([1,2],
    [np.mean(SE_fano_boot_SML),np.mean(SE_fano_boot_RF)],
    yerr=[SML_error,RF_error],
    color=['grey','orange'])

ax.set_ylabel('Fano factor')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.1/0.2','RF'])

if save_figures:
    plt.savefig(fig_dir+'Figure2-Suppl-2-SG-mean-matched-SMLvsRF.svg')

SMLvsRF_diff = SE_fano_boot_SML - SE_fano_boot_RF
N_RFa = int(np.mean(N_SML).round())
p_value_SMLvsRF = bootstrapped_p_for_sampled(SE_fano_boot_SML,
                                              SE_fano_boot_RF,
                                              np.mean(SE_fano_boot_SML), 
                                              np.mean(SE_fano_boot_RF),N_RFa)

print('Fano factor for SML: ' + str(np.mean(SE_fano_boot_SML))+ ' +/- ' + str(SML_error[0]) + ' ' + str(SML_error[1]))
print('Fano factor for RF: ' + str(np.mean(SE_fano_boot_RF))+ ' +/- ' + str(RF_error[0]) + ' ' + str(RF_error[1]))
print('p :', str(p_value_SMLvsRF))