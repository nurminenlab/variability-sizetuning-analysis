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

save_figures = False

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
        vr_LAR[i,:] = vr_matrix[-1,first_tp:last_tp]
    else:
        mn_SML[i,:] = mn_matrix[1,first_tp:last_tp]
        mn_RF[i,:]  = mn_matrix[2,first_tp:last_tp]
        mn_LAR[i,:] = mn_matrix[-1,first_tp:last_tp]
        vr_SML[i,:] = vr_matrix[1,first_tp:last_tp]
        vr_RF[i,:]  = vr_matrix[2,first_tp:last_tp]
        vr_LAR[i,:] = vr_matrix[-1,first_tp:last_tp]

count_bins = np.arange(0,30,1)

FF_SML = np.mean(vr_SML / (mn_SML + eps),axis=1)
FF_RF  = np.mean(vr_RF / (mn_RF + eps),axis=1)
FF_LAR  = np.mean(vr_LAR / (mn_LAR + eps),axis=1)

mn_SML = np.mean(mn_SML,axis=1)
mn_RF  = np.mean(mn_RF,axis=1)
mn_LAR = np.mean(mn_LAR,axis=1)

FA_SML, FA_RF, mean_SML, mean_RF  = dalib.mean_match_FA(mn_SML, 
                                                        mn_RF, 
                                                        FF_SML, 
                                                        FF_RF, 
                                                        count_bins)
count_bins = np.arange(0,60,1)
FA_RF2, FA_LAR, mean_RF2, mean_LAR  = dalib.mean_match_FA(mn_RF, 
                                                        mn_LAR, 
                                                        FF_RF, 
                                                        FF_LAR, 
                                                        count_bins)


means_SMLRF = np.array([np.mean(mean_SML/0.1), np.mean(mean_RF/0.1)])
SEs_SMLRF    = np.array([np.std(mean_SML/0.1)/np.sqrt(len(mean_SML)), 
                        np.std(mean_RF/0.1)/np.sqrt(len(mean_RF))])

means_RFLAR = np.array([np.mean(mean_RF2/0.1), np.mean(mean_LAR/0.1)])
SEs_RFLAR    = np.array([np.std(mean_RF2/0.1)/np.sqrt(len(mean_RF2)), 
                        np.std(mean_LAR/0.1)/np.sqrt(len(mean_LAR))])

FFs = np.array([np.mean(FA_SML), np.mean(FA_RF)])
ffSEs = np.array([np.std(FA_SML)/np.sqrt(len(FA_SML)), 
                  np.std(FA_RF)/np.sqrt(len(FA_RF))])

FFs_RFLAR = np.array([np.mean(FA_RF2), np.mean(FA_LAR)])
ffSEs_RFLAR = np.array([np.std(FA_RF2)/np.sqrt(len(FA_LAR)), 
                  np.std(FA_RF2)/np.sqrt(len(FA_LAR))])

plt.figure(1)
ax = plt.subplot(2,2,1)
ax.bar([0,1], means_SMLRF, yerr=SEs_SMLRF, color=['grey','orange'], width=0.5)
ax = plt.subplot(2,2,2)
ax.bar([0,1], means_RFLAR, yerr=SEs_RFLAR, color=['orange','blue'], width=0.5)

ax = plt.subplot(2,2,3)
plt.bar([0,1], FFs, yerr=ffSEs_RFLAR, color=['grey','orange'], width=0.5)
ax = plt.subplot(2,2,4)
plt.bar([0,1], FFs_RFLAR, yerr=ffSEs_RFLAR, color=['orange','blue'], width=0.5)