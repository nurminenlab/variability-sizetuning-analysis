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
    nearSUR4_idx = np.argmin(np.abs(4*RF_diam - diam))
    RFdiam_idx   = np.argmin(np.abs(RF_diam - diam))
    
    return RFdiam_idx, nearSUR2_idx, nearSUR4_idx


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
        # mean responses 
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

fano_RFa, fano_SR2, mean_RFa, mean_SR2, fano_boot_RFa, fano_boot_SR2,bmeans_RFa, bmeans_SR2, N_RFa, N_SR2 = dalib.PSTH_meanmatch_twopopulations(mn_RFa, 
                                                                                                                                  mn_SR2, 
                                                                                                                                  vr_RFa, 
                                                                                                                                  vr_SR2, 
                                                                                                                                  count_bins,
                                                                                                                                  1000)

fano_RFa_new, fano_SR3, mean_RFa_new, mean_SR3, fano_boot_RFa_new, fano_boot_SR3,bmeans_RFa_new, bmeans_SR3, N_RFa_new, N_SR3 = dalib.PSTH_meanmatch_twopopulations(mn_RFa, 
                                                                                                                                                  mn_SR3, 
                                                                                                                                                  vr_RFa, 
                                                                                                                                                  vr_SR3, 
                                                                                                                                                  count_bins,
                                                                                                                                                  1000)

fano_RFa_LAR, fano_LAR, mean_RFa_LAR, mean_LAR, fano_boot_RFa_LAR, fano_boot_LAR, bmeans_RFa_LAR, bmeans_LAR, N_RFa_LAR, N_LAR = dalib.PSTH_meanmatch_twopopulations(mn_RFa, 
                                                                                                                                                  mn_LAR, 
                                                                                                                                                  vr_RFa, 
                                                                                                                                                  vr_LAR, 
                                                                                                                                                  count_bins,
                                                                                                                                                  1000)

# RUNS UP TO HERE, NOW PLOTTING!
t1 = 50
t = np.linspace(t1,t1+anal_duration,anal_duration)

fig, axes = plt.subplots(4,3,sharex=True)

# RF vs 2RF
#--------------------------
this_row = 0
this_col = 0
axes[this_row,this_col].plot(t,mean_RFa/0.1,'-',color='grey',label='RF')
axes[this_row,this_col].plot(t,mean_SR2/0.1,'-',color='orange',label='2RF')
axes[this_row,this_col].set_ylim([0,135])
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 0
axes[this_row,this_col].fill_between(t,(mean_RFa/0.1) - np.nanstd(bmeans_RFa/0.1,axis=0),(mean_RFa/0.1) + np.nanstd(bmeans_RFa/0.1,axis=0),color='grey',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_SR2/0.1) - np.nanstd(bmeans_SR2/0.1,axis=0),(mean_SR2/0.1) + np.nanstd(bmeans_SR2/0.1,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].plot(t,mean_RFa/0.1,'-',color='grey') # mean response 
axes[this_row,this_col].plot(t,mean_SR2/0.1,'-',color='orange') # mean response
axes[this_row,this_col].set_ylim([0,135])

this_row = 2
this_col = 0
axes[this_row,this_col].plot(t,fano_RFa,'-',color='grey',label='RF')
axes[this_row,this_col].plot(t,fano_SR2,'-',color='orange',label='2RF')
axes[this_row,this_col].set_ylim([0,5])
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 0
axes[this_row,this_col].fill_between(t,(fano_RFa) - np.nanstd(fano_boot_RFa,axis=0),(fano_RFa) + np.nanstd(fano_boot_RFa,axis=0),color='grey',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_SR2) - np.nanstd(fano_boot_SR2,axis=0),(fano_SR2) + np.nanstd(fano_boot_SR2,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].plot(t,fano_RFa,'-',color='grey',label='RF')
axes[this_row,this_col].plot(t,fano_SR2,'-',color='orange',label='2RF')
axes[this_row,this_col].set_ylim([0,5])

# RF vs 3RF
#--------------------------
this_row = 0
this_col = 1
axes[this_row,this_col].plot(t,mean_RFa_new/0.1,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,mean_SR3/0.1,'-',color='blue',label='2RF')
axes[this_row,this_col].set_ylim([0,135])
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 1
axes[this_row,this_col].fill_between(t,(mean_RFa_new/0.1) - np.nanstd(bmeans_RFa_new/0.1,axis=0),(mean_RFa_new/0.1) + np.nanstd(bmeans_RFa_new/0.1,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_SR3/0.1) - np.nanstd(bmeans_SR3/0.1,axis=0),(mean_SR3/0.1) + np.nanstd(bmeans_SR3/0.1,axis=0),color='blue',alpha=0.5)
axes[this_row,this_col].plot(t,mean_RFa_new/0.1,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,mean_SR3/0.1,'-',color='blue',label='2RF')
axes[this_row,this_col].set_ylim([0,135])

this_row = 2
this_col = 1
axes[this_row,this_col].plot(t,fano_RFa_new,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,fano_SR3,'-',color='blue',label='3RF')
axes[this_row,this_col].set_ylim([0,5])
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 1
axes[this_row,this_col].fill_between(t,(fano_RFa_new) - np.nanstd(fano_boot_RFa_new,axis=0),(fano_RFa_new) + np.nanstd(fano_boot_RFa_new,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_SR3) - np.nanstd(fano_boot_SR3,axis=0),(fano_SR3) + np.nanstd(fano_boot_SR3,axis=0),color='blue',alpha=0.5)
axes[this_row,this_col].plot(t,fano_RFa_new,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,fano_SR3,'-',color='blue',label='3RF')
axes[this_row,this_col].set_ylim([0,5])


# RF vs 26
#--------------------------
this_row = 0
this_col = 2
axes[this_row,this_col].plot(t,mean_RFa_LAR/0.1,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,mean_LAR/0.1,'-',color='blue',label='2RF')
axes[this_row,this_col].set_ylim([0,135])
axes[this_row,this_col].set_ylabel('Firing rate (Hz)')

this_row = 1
this_col = 2
axes[this_row,this_col].fill_between(t,(mean_RFa_LAR/0.1) - np.nanstd(bmeans_RFa_LAR/0.1,axis=0),(mean_RFa_LAR/0.1) + np.nanstd(bmeans_RFa_LAR/0.1,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].fill_between(t,(mean_LAR/0.1) - np.nanstd(bmeans_LAR/0.1,axis=0),(mean_LAR/0.1) + np.nanstd(bmeans_LAR/0.1,axis=0),color='blue',alpha=0.5)
axes[this_row,this_col].plot(t,mean_RFa_LAR/0.1,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,mean_LAR/0.1,'-',color='blue',label='2RF')
axes[this_row,this_col].set_ylim([0,135])

this_row = 2
this_col = 2
axes[this_row,this_col].plot(t,fano_RFa_LAR,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,fano_LAR,'-',color='blue',label='3RF')
axes[this_row,this_col].set_ylim([0,5])
axes[this_row,this_col].set_ylabel('Fano factor')

this_row = 3
this_col = 2
axes[this_row,this_col].fill_between(t,(fano_RFa_LAR) - np.nanstd(fano_boot_RFa_LAR,axis=0),(fano_RFa_LAR) + np.nanstd(fano_boot_RFa_LAR,axis=0),color='orange',alpha=0.5)
axes[this_row,this_col].fill_between(t,(fano_LAR) - np.nanstd(fano_boot_LAR,axis=0),(fano_LAR) + np.nanstd(fano_boot_LAR,axis=0),color='blue',alpha=0.5)
axes[this_row,this_col].plot(t,fano_RFa_LAR,'-',color='orange',label='RF')
axes[this_row,this_col].plot(t,fano_LAR,'-',color='blue',label='3RF')
axes[this_row,this_col].set_ylim([0,5])


if save_figures:
    plt.savefig(fig_dir+'Figure2-Suppl-2-SG-mean-matched-PSTHs-nearSURROUND.svg')

plt.figure(2)
ax = plt.subplot(2,2,1)

SE_mean_boot_RFa = np.nanmean(bmeans_RFa,axis=1)
SE_mean_boot_SR2 = np.nanmean(bmeans_SR2,axis=1)
# Compute the confidence interval
alpha = 0.001
RFa_lower_bound = np.percentile(SE_mean_boot_RFa, 100 * alpha / 2)
RFa_upper_bound = np.percentile(SE_mean_boot_RFa, 100 * (1 - alpha / 2))
SR2_lower_bound = np.percentile(SE_mean_boot_SR2, 100 * alpha / 2)
SR2_upper_bound = np.percentile(SE_mean_boot_SR2, 100 * (1 - alpha / 2))
# Compute the error bars
RFa_error = np.array([np.mean(SE_mean_boot_RFa) - RFa_lower_bound, RFa_upper_bound - np.mean(SE_mean_boot_RFa)])
SR2_error = np.array([np.mean(SE_mean_boot_SR2) - SR2_lower_bound, SR2_upper_bound - np.mean(SE_mean_boot_SR2)])

ax.bar([1,2],
    [np.mean(SE_mean_boot_RFa/0.1),np.mean(SE_mean_boot_SR2/0.1)],
    yerr=[RFa_error/0.1,SR2_error/0.1],
    color=['grey','orange'])
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','2RF'])
ax.set_ylabel('Firing rate (Hz)')

ax = plt.subplot(2,2,2)

SE_mean_boot_RFa_new = np.nanmean(bmeans_RFa_new,axis=1)
SE_mean_boot_SR3 = np.nanmean(bmeans_SR3,axis=1)
# Compute the confidence interval
alpha = 0.001
RFa_new_lower_bound = np.percentile(SE_mean_boot_RFa_new, 100 * alpha / 2)
RFa_new_upper_bound = np.percentile(SE_mean_boot_RFa_new, 100 * (1 - alpha / 2))
SR3_lower_bound = np.percentile(SE_mean_boot_SR3, 100 * alpha / 2)
SR3_upper_bound = np.percentile(SE_mean_boot_SR3, 100 * (1 - alpha / 2))
# Compute the error bars
RFa_new_error = np.array([np.mean(SE_mean_boot_RFa_new) - RFa_new_lower_bound, 
                          RFa_new_upper_bound - np.mean(SE_mean_boot_RFa_new)])
SR3_new_error = np.array([np.mean(SE_mean_boot_SR3) - SR3_lower_bound, 
                          SR3_upper_bound - np.mean(SE_mean_boot_SR3)])

ax.bar([1,2],
    [np.mean(SE_mean_boot_RFa_new/0.1),np.mean(SE_mean_boot_SR3/0.1)],
    yerr=[RFa_new_error/0.1,SR3_new_error/0.1],
    color=['orange','blue'])
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','4RF'])
#ax.set_ylabel('Firing rate (Hz)')

ax = plt.subplot(2,2,3)

SE_fano_boot_RFa = np.nanmean(fano_boot_RFa,axis=1)
SE_fano_boot_SR2 = np.nanmean(fano_boot_SR2,axis=1)
# Compute the confidence interval
alpha = 0.001
RFa_lower_bound = np.percentile(SE_fano_boot_RFa, 100 * alpha / 2)
RFa_upper_bound = np.percentile(SE_fano_boot_RFa, 100 * (1 - alpha / 2))
SR2_lower_bound = np.percentile(SE_fano_boot_SR2, 100 * alpha / 2)
SR2_upper_bound = np.percentile(SE_fano_boot_SR2, 100 * (1 - alpha / 2))
# Compute the error bars
RFa_error = [np.mean(SE_fano_boot_RFa) - RFa_lower_bound, RFa_upper_bound - np.mean(SE_fano_boot_RFa)]
SR2_error = [np.mean(SE_fano_boot_SR2) - SR2_lower_bound, SR2_upper_bound - np.mean(SE_fano_boot_SR2)]

ax.bar([1,2],
    [np.mean(SE_fano_boot_RFa),np.mean(SE_fano_boot_SR2)],
    yerr=[RFa_error,SR2_error],
    color=['grey','orange'])

ax.set_ylabel('Fano factor')
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','2RF'])


ax = plt.subplot(2,2,4)

SE_fano_boot_RFa_new = np.nanmean(fano_boot_RFa_new,axis=1)
SE_fano_boot_SR3 = np.nanmean(fano_boot_SR3,axis=1)
# Compute the confidence interval
alpha = 0.001
RFa_new_lower_bound = np.percentile(SE_fano_boot_RFa_new, 100 * alpha / 2)
RFa_new_upper_bound = np.percentile(SE_fano_boot_RFa_new, 100 * (1 - alpha / 2))
SR3_lower_bound = np.percentile(SE_fano_boot_SR3, 100 * alpha / 2)
SR3_upper_bound = np.percentile(SE_fano_boot_SR3, 100 * (1 - alpha / 2))
# Compute the error bars
RFa_new_error = [np.mean(SE_fano_boot_RFa_new) - RFa_new_lower_bound, 
             RFa_new_upper_bound - np.mean(SE_fano_boot_RFa_new)]
SR3_error = [np.mean(SE_fano_boot_SR3) - SR3_lower_bound, 
             SR3_upper_bound - np.mean(SE_fano_boot_SR3)]


ax.bar([1,2],
    [np.mean(SE_fano_boot_RFa_new),np.mean(SE_fano_boot_SR3)],
    yerr=[RFa_new_error,SR3_error],
    color=['orange','blue'])

ax.set_xticks([1,2])
ax.set_xticklabels(['RF','4RF'])

if save_figures:
    plt.savefig(fig_dir+'Figure2-Suppl-2-SG-mean-matched-nearSURROUND.svg')

FF_RF  = np.nanmean(vr_RFa / mn_RFa, axis=1)
FF_RF2 = np.nanmean(vr_SR2 / mn_SR2, axis=1)

RFvs2RF_diff = SE_fano_boot_RFa - SE_fano_boot_SR2
N_RFa = int(np.mean(N_RFa).round())
p_value_RFvs2RF = bootstrapped_p_for_sampled(SE_fano_boot_RFa,
                                              SE_fano_boot_SR2,
                                              np.mean(SE_fano_boot_RFa), 
                                              np.mean(SE_fano_boot_SR2),N_RFa)

RFvs3RF_diff = SE_fano_boot_RFa_new - SE_fano_boot_SR3
N_RFa_new = int(np.mean(N_RFa_new).round())
p_value_RFvs3RF = bootstrapped_p_for_sampled(SE_fano_boot_RFa_new,
                                             SE_fano_boot_SR3,
                                             np.mean(SE_fano_boot_RFa_new),
                                             np.mean(SE_fano_boot_SR3),N_RFa_new)


plt.figure(3)
ax = plt.subplot(2,2,1)

SE_mean_boot_RFa_LAR = np.nanmean(bmeans_RFa_LAR,axis=1)
SE_mean_boot_LAR = np.nanmean(bmeans_LAR,axis=1)
# Compute the confidence interval
alpha = 0.001
RFa_LAR_lower_bound = np.percentile(SE_mean_boot_RFa_LAR, 100 * alpha / 2)
RFa_LAR_upper_bound = np.percentile(SE_mean_boot_RFa_LAR, 100 * (1 - alpha / 2))
LAR_lower_bound = np.percentile(SE_mean_boot_LAR, 100 * alpha / 2)
LAR_upper_bound = np.percentile(SE_mean_boot_LAR, 100 * (1 - alpha / 2))
# Compute the error bars
RFa_LAR_error = np.array([np.mean(SE_mean_boot_RFa_LAR) - RFa_LAR_lower_bound, RFa_LAR_upper_bound - np.mean(SE_mean_boot_RFa_LAR)])
LAR_error = np.array([np.mean(SE_mean_boot_LAR) - LAR_lower_bound, LAR_upper_bound - np.mean(SE_mean_boot_LAR)])

ax.bar([1,2],
    [np.mean(SE_mean_boot_RFa_LAR/0.1),np.mean(SE_mean_boot_LAR/0.1)],
    yerr=[RFa_LAR_error/0.1,LAR_error/0.1],
    color=['grey','orange'])
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','26'])
ax.set_ylabel('Firing rate (Hz)')

ax = plt.subplot(2,2,3)

SE_fano_boot_RFa_LAR = np.nanmean(fano_boot_RFa_LAR,axis=1)
SE_fano_boot_LAR = np.nanmean(fano_boot_LAR,axis=1)
# Compute the confidence interval
alpha = 0.001
RFa_LAR_lower_bound = np.percentile(SE_fano_boot_RFa_LAR, 100 * alpha / 2)
RFa_LAR_upper_bound = np.percentile(SE_fano_boot_RFa_LAR, 100 * (1 - alpha / 2))
LAR_lower_bound = np.percentile(SE_fano_boot_LAR, 100 * alpha / 2)
LAR_upper_bound = np.percentile(SE_fano_boot_LAR, 100 * (1 - alpha / 2))
# Compute the error bars
RFa_LAR_error = [np.mean(SE_fano_boot_RFa_LAR) - RFa_LAR_lower_bound, RFa_LAR_upper_bound - np.mean(SE_fano_boot_RFa_LAR)]
LAR_error = [np.mean(SE_fano_boot_LAR) - LAR_lower_bound, LAR_upper_bound - np.mean(SE_fano_boot_LAR)]

ax.bar([1,2],
    [np.mean(SE_fano_boot_RFa_LAR),np.mean(SE_fano_boot_LAR)],
    yerr=[RFa_LAR_error,LAR_error],
    color=['grey','orange'])

ax.set_ylabel('Fano factor')
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','26'])

if save_figures:
    plt.savefig(fig_dir+'Figure2-Suppl-2-SG-RFvs26-mean-matched-nearSURROUND.svg')

RFvs26_diff = SE_fano_boot_RFa_LAR - SE_fano_boot_LAR
N_RFa_LAR = int(np.mean(N_RFa_LAR).round())
p_value_RFvs26 = bootstrapped_p_for_sampled(SE_fano_boot_RFa_LAR,
                                              SE_fano_boot_LAR,
                                              np.mean(SE_fano_boot_RFa_LAR), 
                                              np.mean(SE_fano_boot_LAR),N_RFa_LAR)


print('Fano factor for RF: ' + str(np.mean(SE_fano_boot_RFa))+ ' +/- ' + str(RFa_error[0]) + ' ' + str(RFa_error[1]))
print('Fano factor for 2RF: ' + str(np.mean(SE_fano_boot_SR2))+ ' +/- ' + str(SR2_error[0]) + ' ' + str(SR2_error[1]))
print('p :', str(p_value_RFvs2RF))
print('####################')
print('Fano factor for RF: ' + str(np.mean(SE_fano_boot_RFa_new))+ ' +/- ' + str(RFa_new_error[0]) + ' ' + str(RFa_new_error[1]))
print('Fano factor for 4RF: ' + str(np.mean(SE_fano_boot_SR3))+ ' +/- ' + str(SR3_error[0]) + ' ' + str(SR3_error[1]))
print('p :', str(p_value_RFvs3RF))
print('####################')
print('Fano factor for RF: ' + str(np.mean(SE_fano_boot_RFa_LAR))+ ' +/- ' + str(RFa_LAR_error[0]) + ' ' + str(RFa_LAR_error[1]))
print('Fano factor for 26: ' + str(np.mean(SE_fano_boot_LAR))+ ' +/- ' + str(LAR_error[0]) + ' ' + str(LAR_error[1]))
print('p :', str(p_value_RFvs26))
