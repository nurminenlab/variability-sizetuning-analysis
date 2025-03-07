import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
#import pdb
import statsmodels.api as sm

import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
from scipy.optimize import basinhopping, curve_fit

import scipy.stats as sts

# this way of computing the mean firing-rate functions is cumbersome but we do it this way for consistency with the other scripts
save_figures = True

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
data_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/2-PrecomputedAnalysis/'

nconds = 4

SG_netvariance = np.load(data_dir+'netvariance_all_IG.npy')
SG_meanresponses = np.load(data_dir+'mean_response_all_IG.npy')

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)

diams = np.array(list(diams_data.keys())).round(1)
del(diams_data)

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

def fit_mean_response(meanR,diam):
    addone = 0
    SMLidx = 0    
    if np.isnan(meanR[0]):
        meanR = meanR[1:]   
        diam = diam[1:]
        addone = 1

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
    
    return SMLidx+addone, RFdiam_idx+addone, nearSUR2_idx+addone, nearSUR3_idx+addone


# SG
SG_match_mean_SML  = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_mean_RF   = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_mean_SUR2 = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_mean_SUR3 = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_mean_LAR  = np.nan * np.ones((SG_meanresponses.shape[0],))

SG_match_FA_SML   = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_FA_RF    = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_FA_SUR2  = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_FA_SUR3  = np.nan * np.ones((SG_meanresponses.shape[0],))
SG_match_FA_LAR   = np.nan * np.ones((SG_meanresponses.shape[0],))

for i in range(SG_meanresponses.shape[0]):
    SMLidx, RF_idx, SUR2_idx, SUR3_idx = fit_mean_response(SG_meanresponses[i,:], diams)
    SG_match_mean_SML[i]  = SG_meanresponses[i,SMLidx]
    SG_match_mean_RF[i]   = SG_meanresponses[i,RF_idx]
    SG_match_mean_SUR2[i] = SG_meanresponses[i,SUR2_idx]
    SG_match_mean_SUR3[i] = SG_meanresponses[i,SUR3_idx]
    SG_match_mean_LAR[i]  = SG_meanresponses[i,-1]

    SG_match_FA_SML[i]  = SG_netvariance[i,SMLidx]
    SG_match_FA_RF[i]   = SG_netvariance[i,RF_idx]
    SG_match_FA_SUR2[i] = SG_netvariance[i,SUR2_idx]
    SG_match_FA_SUR3[i] = SG_netvariance[i,SUR3_idx]
    SG_match_FA_LAR[i]  = SG_netvariance[i,-1]


count_bins = np.arange(0,150,5)
# 0.1 vs RF
FA_SML, FA_RF, mean_SML, mean_RF, N_SML, STD_SML, STD_RF, FA_noboot_SML, FA_noboot_RF = dalib.mean_match_FA(SG_match_mean_SML, 
                                                                        SG_match_mean_RF, 
                                                                        SG_match_FA_SML, 
                                                                        SG_match_FA_RF, 
                                                                        count_bins)

# RF vs 2RF
FA_RF2, FA_SUR2, mean_RF2, mean_SUR2, N_SUR2, STD_RF2, STD_SUR2, FA_noboot_RF2, FA_noboot_SUR2  = dalib.mean_match_FA(SG_match_mean_RF, 
                                                                        SG_match_mean_SUR2, 
                                                                        SG_match_FA_RF, 
                                                                        SG_match_FA_SUR2, 
                                                                        count_bins)

# RF vs 3RF
FA_RF3, FA_SUR3, mean_RF3, mean_SUR3, N_SUR3, STD_RF3, STD_SUR3, FA_noboot_RF3, FA_noboot_SUR3  = dalib.mean_match_FA(SG_match_mean_RF, 
                                                                    SG_match_mean_SUR3, 
                                                                    SG_match_FA_RF, 
                                                                    SG_match_FA_SUR3, 
                                                                    count_bins)

# RF vs 26
FA_RF_LAR, FA_LAR, mean_RF_LAR, mean_LAR, N_LAR, STD_RF_LAR, STD_LAR, FA_noboot_RF_LAR, FA_noboot_LAR  = dalib.mean_match_FA(SG_match_mean_RF, 
                                                                        SG_match_mean_LAR, 
                                                                        SG_match_FA_RF, 
                                                                        SG_match_FA_LAR, 
                                                                        count_bins)


plt.figure(1)
# Compute the confidence interval for the mean
# do stats and plot
# 0.1 vs RF
#--------------------------
cond = 1
alpha = 0.001

# mean
CI_SML_lower_bound = np.percentile(mean_SML, 100 * alpha / 2)
CI_SML_upper_bound = np.percentile(mean_SML, 100 * (1 - alpha / 2))
CI_RF_lower_bound  = np.percentile(mean_RF, 100 * alpha / 2)
CI_RF_upper_bound  = np.percentile(mean_RF, 100 * (1 - alpha / 2))

""" SML_error = np.nanmean(np.array([np.mean(mean_SML) - CI_SML_lower_bound, CI_SML_upper_bound - np.mean(mean_SML)]))
RF_error  = np.nanmean(np.array([np.mean(mean_RF) - CI_RF_lower_bound, CI_RF_upper_bound - np.mean(mean_RF)])) """


ax = plt.subplot(2,nconds,cond)
ax.bar([1,2],
    np.array([np.nanmean(mean_SML),np.nanmean(mean_RF)]),
    yerr=np.array([STD_SML,STD_RF])/np.sqrt(int(np.mean(N_SML))),
    color=['grey','orange'])

ax.set_xticks([1,2])
ax.set_xticklabels(['',''])
ax.set_ylim(0, 85)
ax.set_ylabel('Mean-matched firing-rate')

# FA
alpha = 0.16
CI_SML_lower_bound = np.percentile(FA_SML, 100 * alpha / 2)
CI_SML_upper_bound = np.percentile(FA_SML, 100 * (1 - alpha / 2))
CI_RF_lower_bound  = np.percentile(FA_RF, 100 * alpha / 2)
CI_RF_upper_bound  = np.percentile(FA_RF, 100 * (1 - alpha / 2))

""" SML_error = np.mean(np.array([np.mean(FA_SML) - CI_SML_lower_bound, CI_SML_upper_bound - np.mean(FA_SML)]))
RF_error  = np.mean(np.array([np.mean(FA_RF) - CI_RF_lower_bound, CI_RF_upper_bound - np.mean(FA_RF)])) """

SML_error = np.std(FA_noboot_SML)/np.sqrt(FA_noboot_SML.shape[0])
RF_error = np.std(FA_noboot_RF)/np.sqrt(FA_noboot_RF.shape[0])

ax = plt.subplot(2,nconds,cond+nconds)
ax.bar([1,2],
    np.array([np.mean(FA_SML),np.mean(FA_RF)]),
    yerr=[SML_error,RF_error],
    color=['grey','orange'])

ax.set_ylim(0, 1)
ax.set_ylabel('Shared variance')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.1/0.2','RF'])

p_SML = dalib.bootstrapped_p_for_sampled(FA_noboot_SML,
                                         FA_noboot_RF,
                                         np.mean(FA_noboot_SML),
                                         np.mean(FA_noboot_RF),
                                         FA_noboot_SML.shape[0])

print('\n0.1/0.2 vs RF')
print('G: 0.1/0.2'+ ' mean: ' + str(np.mean(FA_SML)) + ' se: ' + str(SML_error))
print('G: RF'+ ' mean: ' + str(np.mean(FA_RF)) + ' se: ' + str(RF_error)) 
print('p-value: ', p_SML)

# RF vs 2RF
#--------------------------
cond = 2
alpha = 0.001

# mean
CI_RF2_lower_bound = np.percentile(mean_RF2, 100 * alpha / 2)
CI_RF2_upper_bound = np.percentile(mean_RF2, 100 * (1 - alpha / 2))
CI_SUR2_lower_bound  = np.percentile(mean_SUR2, 100 * alpha / 2)
CI_SUR2_upper_bound  = np.percentile(mean_SUR2, 100 * (1 - alpha / 2))

""" RF2_error = np.nanmean(np.array([np.mean(mean_RF2) - CI_RF2_lower_bound, CI_RF2_upper_bound - np.mean(mean_RF2)]))
SUR2_error  = np.nanmean(np.array([np.mean(mean_SUR2) - CI_SUR2_lower_bound, CI_SUR2_upper_bound - np.mean(mean_SUR2)])) """

ax = plt.subplot(2,nconds,cond)
ax.bar([1,2],
    np.array([np.nanmean(mean_RF2),np.nanmean(mean_SUR2)]),
    yerr=np.array([STD_RF2,STD_SUR2])/np.sqrt(int(np.mean(N_SUR2))),
    color=['grey','orange'])
ax.set_xticks([1,2])
ax.set_xticklabels(['',''])
ax.set_ylim(0, 85)

# FA
alpha = 0.16
CI_RF2_lower_bound = np.percentile(FA_RF2, 100 * alpha / 2)
CI_RF2_upper_bound = np.percentile(FA_RF2, 100 * (1 - alpha / 2))
CI_SUR2_lower_bound  = np.percentile(FA_SUR2, 100 * alpha / 2)
CI_SUR2_upper_bound  = np.percentile(FA_SUR2, 100 * (1 - alpha / 2))

""" RF2_error  = np.mean(np.array([np.mean(FA_RF2) - CI_RF2_lower_bound, CI_RF2_upper_bound - np.mean(FA_RF2)]))
SUR2_error = np.mean(np.array([np.mean(FA_SUR2) - CI_SUR2_lower_bound, CI_SUR2_upper_bound - np.mean(FA_SUR2)])) """

RF2_error = np.std(FA_noboot_RF2)/np.sqrt(FA_noboot_RF2.shape[0])
SUR2_error = np.std(FA_noboot_SUR2)/np.sqrt(FA_noboot_SUR2.shape[0])

ax = plt.subplot(2,nconds,cond+nconds)
ax.bar([1,2],
    np.array([np.mean(FA_RF2),np.mean(FA_SUR2)]),
    yerr=[RF2_error,SUR2_error],
    color=['grey','orange'])

ax.set_ylim(0, 1)
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','2*RF'])

p_SUR2 = dalib.bootstrapped_p_for_sampled(FA_noboot_RF2,
                                          FA_noboot_SUR2,
                                          np.mean(FA_noboot_RF2),
                                          np.mean(FA_noboot_SUR2),
                                          FA_noboot_RF2.shape[0])

print('\nRF vs 2*RF')
print('G: RF'+ ' mean: ' + str(np.mean(FA_RF2)) + ' se: ' + str(RF2_error))
print('G: 2*RF'+ ' mean: ' + str(np.mean(FA_SUR2)) + ' se: ' + str(SUR2_error)) 
print('p-value: ', p_SUR2)


# RF vs 3RF
#--------------------------
cond = 3
alpha = 0.001

# mean
CI_RF3_lower_bound = np.percentile(mean_RF3, 100 * alpha / 2)
CI_RF3_upper_bound = np.percentile(mean_RF3, 100 * (1 - alpha / 2))
CI_SUR3_lower_bound  = np.percentile(mean_SUR3, 100 * alpha / 2)
CI_SUR3_upper_bound  = np.percentile(mean_SUR3, 100 * (1 - alpha / 2))

""" RF3_error = np.nanmean(np.array([np.mean(mean_RF3) - CI_RF3_lower_bound, CI_RF3_upper_bound - np.mean(mean_RF3)]))
SUR3_error = np.nanmean(np.array([np.mean(mean_SUR3) - CI_SUR3_lower_bound, CI_SUR3_upper_bound - np.mean(mean_SUR3)])) """



ax = plt.subplot(2,nconds,cond)
ax.bar([1,2],
    np.array([np.nanmean(mean_RF3),np.nanmean(mean_SUR3)]),
    yerr=np.array([STD_RF3,STD_SUR3])/np.sqrt(int(np.mean(N_SUR3))),
    color=['grey','orange'])
ax.set_xticks([1,2])
ax.set_xticklabels(['',''])
ax.set_ylim(0, 85)

# FA
alpha = 0.16
CI_RF3_lower_bound = np.percentile(FA_RF3, 100 * alpha / 2)
CI_RF3_upper_bound = np.percentile(FA_RF3, 100 * (1 - alpha / 2))
CI_SUR3_lower_bound  = np.percentile(FA_SUR3, 100 * alpha / 2)
CI_SUR3_upper_bound  = np.percentile(FA_SUR3, 100 * (1 - alpha / 2))

""" RF3_error  = np.mean(np.array([np.mean(FA_RF3) - CI_RF3_lower_bound, CI_RF3_upper_bound - np.mean(FA_RF3)]))
SUR3_error = np.mean(np.array([np.mean(FA_SUR3) - CI_SUR3_lower_bound, CI_SUR3_upper_bound - np.mean(FA_SUR3)])) """

RF3_error = np.std(FA_noboot_RF3)/np.sqrt(FA_noboot_RF3.shape[0])
SUR3_error = np.std(FA_noboot_SUR3)/np.sqrt(FA_noboot_SUR3.shape[0])

ax = plt.subplot(2,nconds,cond+nconds)
ax.bar([1,2],
    np.array([np.mean(FA_RF3),np.mean(FA_SUR3)]),
    yerr=[RF3_error,SUR3_error],
    color=['grey','orange'])

ax.set_ylim(0, 1)
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','3 * RF'])

p_SUR3 = dalib.bootstrapped_p_for_sampled(FA_noboot_RF3,
                                          FA_noboot_SUR3,
                                          np.mean(FA_noboot_RF3),
                                          np.mean(FA_noboot_SUR3),
                                          FA_noboot_RF3.shape[0])

print('\nRF vs 3*RF')
print('G: RF'+ ' mean: ' + str(np.mean(FA_RF3)) + ' se: ' + str(RF3_error))
print('G: 3*RF'+ ' mean: ' + str(np.mean(FA_SUR3)) + ' se: ' + str(SUR3_error)) 
print('p-value: ', p_SUR3)

# RF vs 26
#--------------------------
cond = 4
alpha = 0.001

# mean
CI_RF_LAR_lower_bound = np.percentile(mean_RF_LAR, 100 * alpha / 2)
CI_RF_LAR_upper_bound = np.percentile(mean_RF_LAR, 100 * (1 - alpha / 2))
CI_LAR_lower_bound  = np.percentile(mean_LAR, 100 * alpha / 2)
CI_LAR_upper_bound  = np.percentile(mean_LAR, 100 * (1 - alpha / 2))

RF_LAR_error = np.nanmean(np.array([np.mean(mean_RF_LAR) - CI_RF_LAR_lower_bound, CI_RF_LAR_upper_bound - np.mean(mean_RF_LAR)]))
LAR_error = np.nanmean(np.array([np.mean(mean_LAR) - CI_LAR_lower_bound, CI_LAR_upper_bound - np.mean(mean_LAR)]))

ax = plt.subplot(2,nconds,cond)
ax.bar([1,2],
    np.array([np.nanmean(mean_RF_LAR),np.nanmean(mean_LAR)]),
    yerr=np.array([STD_RF_LAR,STD_LAR])/np.sqrt(int(np.mean(N_LAR))),
    color=['grey','orange'])
ax.set_xticks([1,2])
ax.set_xticklabels(['',''])
ax.set_ylim(0, 85)

# FA
alpha = 0.001
CI_RF_LAR_lower_bound = np.percentile(FA_RF_LAR, 100 * alpha / 2)
CI_RF_LAR_upper_bound = np.percentile(FA_RF_LAR, 100 * (1 - alpha / 2))
CI_LAR_lower_bound  = np.percentile(FA_LAR, 100 * alpha / 2)
CI_LAR_upper_bound  = np.percentile(FA_LAR, 100 * (1 - alpha / 2))

""" RF_LAR_error  = np.mean(np.array([np.mean(FA_RF_LAR) - CI_RF_LAR_lower_bound, CI_RF_LAR_upper_bound - np.mean(FA_RF_LAR)]))
LAR_error = np.mean(np.array([np.mean(FA_LAR) - CI_LAR_lower_bound, CI_LAR_upper_bound - np.mean(FA_LAR)])) """

RF_LAR_error = np.std(FA_noboot_RF_LAR)/np.sqrt(FA_noboot_RF_LAR.shape[0])
LAR_error = np.std(FA_noboot_LAR)/np.sqrt(FA_noboot_LAR.shape[0])

ax = plt.subplot(2,nconds,cond+nconds)
ax.bar([1,2],
    np.array([np.mean(FA_RF_LAR),np.mean(FA_LAR)]),
    yerr=[RF_LAR_error,LAR_error],
    color=['grey','orange'])

ax.set_ylim(0, 1)
ax.set_xticks([1,2])
ax.set_xticklabels(['RF','26'])

p_LAR = dalib.bootstrapped_p_for_sampled(FA_noboot_RF_LAR,
                                          FA_noboot_LAR,
                                          np.mean(FA_noboot_RF_LAR),
                                          np.mean(FA_noboot_LAR),
                                          FA_noboot_RF_LAR.shape[0])

print('\nRF vs 26')
print('G: RF'+ ' mean: ' + str(np.mean(FA_RF_LAR)) + ' se: ' + str(RF_LAR_error))
print('G: 26'+ ' mean: ' + str(np.mean(FA_LAR)) + ' se: ' + str(LAR_error)) 
print('p-value: ', p_LAR)

if save_figures:
    plt.savefig(fig_dir+'IG-FA-meanmatch.svg')
