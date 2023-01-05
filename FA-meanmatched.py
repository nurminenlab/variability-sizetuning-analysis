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
save_figures = False

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'

data_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/'

SG_netvariance = np.load(data_dir+'netvariance_all_SG.npy')
G_netvariance  = np.load(data_dir+'netvariance_all_G.npy')
IG_netvariance = np.load(data_dir+'netvariance_all_IG.npy')

SG_meanresponses = np.load(data_dir+'mean_response_all_SG.npy')
G_meanresponses  = np.load(data_dir+'mean_response_all_G.npy')
IG_meanresponses = np.load(data_dir+'mean_response_all_IG.npy')


# SG
SG_match_mean_SML = np.nan * np.ones((SG_meanresponses.shape[0],1))
SG_match_FA_SML   = np.nan * np.ones((SG_meanresponses.shape[0],1))
SG_match_mean_SML[:,0] = SG_meanresponses[:,1]
SG_match_FA_SML[:,0] = SG_netvariance[:,1]

SG_match_mean_RF = np.nan * np.ones((SG_meanresponses.shape[0],1))
SG_match_FA_RF   = np.nan * np.ones((SG_meanresponses.shape[0],1))
SG_match_mean_RF[:,0] = SG_meanresponses[:,2]
SG_match_FA_RF[:,0] = SG_netvariance[:,2]

SG_match_mean_LAR = np.nan * np.ones((SG_meanresponses.shape[0],1))
SG_match_FA_LAR   = np.nan * np.ones((SG_meanresponses.shape[0],1))
SG_match_mean_LAR[:,0] = SG_meanresponses[:,-1]
SG_match_FA_LAR[:,0] = SG_netvariance[:,-1]

# G
G_match_mean_SML = np.nan * np.ones((G_meanresponses.shape[0],1))
G_match_FA_SML   = np.nan * np.ones((G_meanresponses.shape[0],1))
G_match_mean_SML[:,0] = G_meanresponses[:,1]
G_match_FA_SML[:,0] = G_netvariance[:,1]

G_match_mean_RF = np.nan * np.ones((G_meanresponses.shape[0],1))
G_match_FA_RF   = np.nan * np.ones((G_meanresponses.shape[0],1))
G_match_mean_RF[:,0] = G_meanresponses[:,2]
G_match_FA_RF[:,0] = G_netvariance[:,2]

G_match_mean_LAR = np.nan * np.ones((G_meanresponses.shape[0],1))
G_match_FA_LAR   = np.nan * np.ones((G_meanresponses.shape[0],1))
G_match_mean_LAR[:,0] = G_meanresponses[:,-1]
G_match_FA_LAR[:,0] = G_netvariance[:,-1]

# IG
IG_match_mean_SML = np.nan * np.ones((IG_meanresponses.shape[0],1))
IG_match_FA_SML   = np.nan * np.ones((IG_meanresponses.shape[0],1))
IG_match_mean_SML[:,0] = IG_meanresponses[:,1]
IG_match_FA_SML[:,0] = IG_netvariance[:,1]

IG_match_mean_RF = np.nan * np.ones((IG_meanresponses.shape[0],1))
IG_match_FA_RF   = np.nan * np.ones((IG_meanresponses.shape[0],1))
IG_match_mean_RF[:,0] = IG_meanresponses[:,2]
IG_match_FA_RF[:,0] = IG_netvariance[:,2]

IG_match_mean_LAR = np.nan * np.ones((IG_meanresponses.shape[0],1))
IG_match_FA_LAR   = np.nan * np.ones((IG_meanresponses.shape[0],1))
IG_match_mean_LAR[:,0] = IG_meanresponses[:,-1]
IG_match_FA_LAR[:,0] = IG_netvariance[:,-1]

count_bins = np.arange(0,150,1)
FA_SML, FA_RF, mean_SML, mean_RF  = dalib.mean_match_FA(SG_match_mean_SML[:,0], 
                                                        SG_match_mean_RF[:,0], 
                                                        SG_match_FA_SML[:,0], 
                                                        SG_match_FA_RF[:,0], 
                                                        count_bins)
# SG
#######################
plt.figure(1)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(FA_SML),np.mean(FA_RF)]),
    yerr=[np.nanstd(FA_SML)/np.sqrt(FA_SML.shape[0]),np.nanstd(FA_RF)/np.sqrt(FA_SML.shape[0])],
    color=['b','r'])

ax.set_ylabel('Shared variance')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])
ax.set_title('SG')

ax = plt.subplot(2,2,2)
ax.bar([1,2],
    np.array([np.mean(mean_SML),np.mean(mean_RF)]),
    yerr=[np.nanstd(mean_SML)/np.sqrt(mean_SML.shape[0]),np.nanstd(mean_RF)/np.sqrt(mean_RF.shape[0])],
    color=['b','r'])

ax.set_ylabel('Firing-rate')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])

print('SG: SML'+ ' mean: ' + str(np.mean(FA_SML)) + ' se: ' + str(np.std(FA_SML)/np.sqrt(FA_SML.shape[0])))
print('SG: RF'+ ' mean: ' + str(np.mean(FA_RF)) + ' se: ' + str(np.std(FA_RF)/np.sqrt(FA_RF.shape[0])))
print('p-value: ', sts.ttest_ind(FA_SML,FA_RF))

if save_figures:
    plt.savefig(fig_dir+'SG_FA_meanmatch_SMLvsRF.svg')


count_bins = np.arange(0,100,3)
FA_RF, FA_LAR, mean_RF, mean_LAR = dalib.mean_match_FA(SG_match_mean_RF, 
                                                            SG_match_mean_LAR, 
                                                            SG_match_FA_RF, 
                                                            SG_match_FA_LAR, 
                                                            count_bins)

plt.figure(2)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(FA_RF),np.mean(FA_LAR)]),
    yerr=[np.nanstd(FA_RF)/np.sqrt(FA_RF.shape[0]),np.nanstd(FA_LAR)/np.sqrt(FA_LAR.shape[0])],
    color=['b','r'])

ax.set_ylabel('Shared variance')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])
ax.set_title('SG')

ax = plt.subplot(2,2,2)
ax.bar([1,2],
    np.array([np.mean(mean_RF),np.mean(mean_LAR)]),
    yerr=[np.nanstd(mean_RF)/np.sqrt(mean_RF.shape[0]),np.nanstd(mean_LAR)/np.sqrt(mean_LAR.shape[0])],
    color=['b','r'])

ax.set_ylabel('Firing-rate')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])

print('SG: RF'+ ' mean: ' + str(np.mean(FA_RF)) + ' se: ' + str(np.std(FA_RF)/np.sqrt(FA_RF.shape[0])))
print('SG: LAR'+ ' mean: ' + str(np.mean(FA_LAR)) + ' se: ' + str(np.std(FA_LAR)/np.sqrt(FA_LAR.shape[0])))
print('p-value: ', sts.ttest_ind(FA_LAR,FA_RF))


if save_figures:
    plt.savefig(fig_dir+'SG_FA_meanmatch_RFvsLAR.svg')
#######################

# G
#######################
count_bins = np.arange(0,100,1)
FA_SML, FA_RF, mean_SML, mean_RF  = dalib.mean_match_FA(G_match_mean_SML[:,0], 
                                                        G_match_mean_RF[:,0], 
                                                        G_match_FA_SML[:,0], 
                                                        G_match_FA_RF[:,0], 
                                                        count_bins)

plt.figure(3)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(FA_SML),np.mean(FA_RF)]),
    yerr=[np.nanstd(FA_SML)/np.sqrt(FA_SML.shape[0]),np.nanstd(FA_RF)/np.sqrt(FA_RF.shape[0])],
    color=['b','r'])

ax.set_ylabel('Shared variance')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])
ax.set_title('G')

ax = plt.subplot(2,2,2)
ax.bar([1,2],
    np.array([np.mean(mean_SML),np.mean(mean_RF)]),
    yerr=[np.nanstd(mean_SML)/np.sqrt(mean_SML.shape[0]),np.nanstd(mean_RF)/np.sqrt(mean_RF.shape[0])],
    color=['b','r'])

ax.set_ylabel('Firing-rate')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])

print('G: SML'+ ' mean: ' + str(np.mean(FA_SML)) + ' se: ' + str(np.std(FA_SML)/np.sqrt(FA_SML.shape[0])))
print('G: RF'+ ' mean: ' + str(np.mean(FA_RF)) + ' se: ' + str(np.std(FA_RF)/np.sqrt(FA_RF.shape[0])))
print('p-value: ', sts.ttest_ind(FA_SML,FA_RF))

if save_figures:
    plt.savefig(fig_dir+'G_FA_meanmatch_SMLvsRF.svg')


count_bins = np.arange(0,100,5)
FA_RF, FA_LAR, mean_RF, mean_LAR = dalib.mean_match_FA(G_match_mean_RF, 
                                                       G_match_mean_LAR, 
                                                       G_match_FA_RF, 
                                                       G_match_FA_LAR, 
                                                       count_bins)


plt.figure(4)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(FA_RF),np.mean(FA_LAR)]),
    yerr=[np.nanstd(FA_RF)/np.sqrt(FA_RF.shape[0]),np.nanstd(FA_LAR)/np.sqrt(FA_LAR.shape[0])],
    color=['b','r'])

ax.set_ylabel('Shared variance')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])
ax.set_title('G')

ax = plt.subplot(2,2,2)
ax.bar([1,2],
    np.array([np.mean(mean_RF),np.mean(mean_LAR)]),
    yerr=[np.nanstd(mean_RF)/np.sqrt(mean_RF.shape[0]),np.nanstd(mean_LAR)/np.sqrt(mean_LAR.shape[0])],
    color=['b','r'])

ax.set_ylabel('Firing-rate')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])

print('G: RF'+ ' mean: ' + str(np.mean(FA_RF)) + ' se: ' + str(np.std(FA_RF)/np.sqrt(FA_RF.shape[0])))
print('G: LAR'+ ' mean: ' + str(np.mean(FA_LAR)) + ' se: ' + str(np.std(FA_LAR)/np.sqrt(FA_LAR.shape[0])))
print('p-value: ', sts.ttest_ind(FA_LAR,FA_RF))

if save_figures:
    plt.savefig(fig_dir+'G_FA_meanmatch_RFvsLAR.svg')
#######################


# IG
#######################
count_bins = np.arange(0,100,1)
FA_SML, FA_RF, mean_SML, mean_RF  = dalib.mean_match_FA(IG_match_mean_SML[:,0], 
                                                        IG_match_mean_RF[:,0], 
                                                        IG_match_FA_SML[:,0], 
                                                        IG_match_FA_RF[:,0], 
                                                        count_bins)

plt.figure(5)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(FA_SML),np.mean(FA_RF)]),
    yerr=[np.nanstd(FA_SML)/np.sqrt(FA_SML.shape[0]),np.nanstd(FA_RF)/np.sqrt(FA_RF.shape[0])],
    color=['b','r'])

ax.set_ylabel('Shared variance')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])
ax.set_title('IG')

ax = plt.subplot(2,2,2)
ax.bar([1,2],
    np.array([np.mean(mean_SML),np.mean(mean_RF)]),
    yerr=[np.nanstd(mean_SML)/np.sqrt(mean_SML.shape[0]),np.nanstd(mean_RF)/np.sqrt(mean_RF.shape[0])],
    color=['b','r'])

ax.set_ylabel('Firing-rate')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.2','0.4'])

print('IG: SML'+ ' mean: ' + str(np.mean(FA_SML)) + ' se: ' + str(np.std(FA_SML)/np.sqrt(FA_SML.shape[0])))
print('IG: RF'+ ' mean: ' + str(np.mean(FA_RF)) + ' se: ' + str(np.std(FA_RF)/np.sqrt(FA_RF.shape[0])))
print('p-value: ', sts.ttest_ind(FA_SML,FA_RF))

if save_figures:
    plt.savefig(fig_dir+'IG_FA_meanmatch_SMLvsRF.svg')

count_bins = np.arange(0,150,1)
FA_RF, FA_LAR, mean_RF, mean_LAR = dalib.mean_match_FA(IG_match_mean_RF, 
                                                            IG_match_mean_LAR, 
                                                            IG_match_FA_RF, 
                                                            IG_match_FA_LAR, 
                                                            count_bins)


plt.figure(6)
ax = plt.subplot(2,2,1)
ax.bar([1,2],
    np.array([np.mean(FA_RF),np.mean(FA_LAR)]),
    yerr=[np.nanstd(FA_RF)/np.sqrt(FA_RF.shape[0]),np.nanstd(FA_LAR)/np.sqrt(FA_LAR.shape[0])],
    color=['b','r'])

ax.set_ylabel('Shared variance')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])
ax.set_title('IG')

ax = plt.subplot(2,2,2)
ax.bar([1,2],
    np.array([np.mean(mean_RF),np.mean(mean_LAR)]),
    yerr=[np.nanstd(mean_RF)/np.sqrt(mean_RF.shape[0]),np.nanstd(mean_LAR)/np.sqrt(mean_LAR.shape[0])],
    color=['b','r'])

ax.set_ylabel('Firing-rate')
ax.set_xticks([1,2])
ax.set_xticklabels(['0.4','26'])

print('IG: RF'+ ' mean: ' + str(np.mean(FA_RF)) + ' se: ' + str(np.std(FA_RF)/np.sqrt(FA_RF.shape[0])))
print('IG: LAR'+ ' mean: ' + str(np.mean(FA_LAR)) + ' se: ' + str(np.std(FA_LAR)/np.sqrt(FA_LAR.shape[0])))
print('p-value: ', sts.ttest_ind(FA_LAR,FA_RF))

if save_figures:
    plt.savefig(fig_dir+'IG_FA_meanmatch_RFvsLAR.svg')

#######################
