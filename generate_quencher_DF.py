import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
sys.path.append('C:/Users/lonurmin/Desktop/code/DataAnalysis')
import data_analysislib as dalib
#import pdb
import statsmodels.api as sm
from statsmodels.formula.api import ols

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
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

# param table
quencher_DF = pd.DataFrame(columns=['FF_sup',
                                    'SI',
                                    'layer',
                                    'FF_sup_magn',
                                    'unit'])

# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100
nboots = 3000
for unit in list(SG_mn_data.keys()):
    # loop diams
    mn_mtrx = SG_mn_data[unit]
    vr_mtrx = SG_vr_data[unit]

    fano       = np.nan * np.ones((mn_mtrx.shape[0]))
    FR         = np.nan * np.ones((mn_mtrx.shape[0]))
    fano_boot  = np.nan * np.ones((nboots,mn_mtrx.shape[0]))

    for stim in range(mn_mtrx.shape[0]):
        fano[stim] = sm.OLS(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window],mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]).fit().params[0]
        FR[stim]   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]
            

        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=nboots)

        # compute bootstrapped fano time-course
        for boot_num in range(mean_PSTH_booted.shape[0]):
            fano_boot[boot_num,stim] = sm.OLS(vari_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window],
                                            mean_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window]).fit().params[0]

        
    # get FF @ RF size
    RFind = np.argmax(FR)

    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,-1],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,-1] - u2 + uc

    delta_FF = fano[RFind] - fano[-1]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup = 'suppresser'
        FF_sup_magn = (fano[-1] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup = 'facilitator'
        FF_sup_magn = (fano[-1] - fano[RFind]) / fano[RFind]
    else:
        FF_sup = 'nonether'
        FF_sup_magn = np.nan
        
    SI = (np.max(FR) - FR[-1]) / np.max(FR)
    para_tmp = {'FF_sup':FF_sup,'SI':SI,'layer':'SG','FF_sup_magn':FF_sup_magn,'unit':unit}
    tmp_df    = pd.DataFrame(para_tmp, index=[indx])
    quencher_DF = quencher_DF.append(tmp_df,sort=True)
    indx += 1


# G
for unit in list(G_mn_data.keys()):
    # loop diams
    mn_mtrx = G_mn_data[unit]
    vr_mtrx = G_vr_data[unit]

    fano       = np.nan * np.ones((mn_mtrx.shape[0]))
    FR         = np.nan * np.ones((mn_mtrx.shape[0]))
    fano_boot  = np.nan * np.ones((nboots,mn_mtrx.shape[0]))

    for stim in range(mn_mtrx.shape[0]):
        fano[stim] = sm.OLS(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window],mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]).fit().params[0]
        FR[stim]   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)        
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]
            

        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=nboots)

        # compute bootstrapped fano time-course
        for boot_num in range(mean_PSTH_booted.shape[0]):
            fano_boot[boot_num,stim] = sm.OLS(vari_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window],
                                            mean_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window]).fit().params[0]




    # get FF @ RF size
    RFind = np.argmax(FR)

    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,-1],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,-1] - u2 + uc

    delta_FF = fano[RFind] - fano[-1]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup = 'suppresser'
        FF_sup_magn = (fano[-1] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup = 'facilitator'
        FF_sup_magn = (fano[-1] - fano[RFind]) / fano[RFind]
    else:
        FF_sup = 'nonether'
        FF_sup_magn = np.nan
        
    SI = (np.max(FR) - FR[-1]) / np.max(FR)
    para_tmp = {'FF_sup':FF_sup,'SI':SI,'layer':'G','FF_sup_magn':FF_sup_magn,'unit':unit}
    tmp_df    = pd.DataFrame(para_tmp, index=[indx])
    quencher_DF = quencher_DF.append(tmp_df,sort=True)
    indx += 1

    
# IG
for unit in list(IG_mn_data.keys()):
    # loop diams
    mn_mtrx = IG_mn_data[unit]
    vr_mtrx = IG_vr_data[unit]

    fano       = np.nan * np.ones((mn_mtrx.shape[0]))
    FR         = np.nan * np.ones((mn_mtrx.shape[0]))
    fano_boot  = np.nan * np.ones((nboots,mn_mtrx.shape[0]))

    for stim in range(mn_mtrx.shape[0]):
        fano[stim] = sm.OLS(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window],mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]).fit().params[0]
        FR[stim]   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)                
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]
            

        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=3000)
        # compute bootstrapped fano time-course
        for boot_num in range(mean_PSTH_booted.shape[0]):
            fano_boot[boot_num,stim] = sm.OLS(vari_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window],
                                            mean_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window]).fit().params[0]


    # get FF @ RF size
    RFind = np.argmax(FR)

    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,-1],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,-1] - u2 + uc

    delta_FF = fano[RFind] - fano[-1]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup = 'suppresser'
        FF_sup_magn = (fano[-1] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup = 'facilitator'
        FF_sup_magn = (fano[-1] - fano[RFind]) / fano[RFind]
    else:
        FF_sup = 'nonether'
        FF_sup_magn = np.nan
        
    SI = (np.max(FR) - FR[-1]) / np.max(FR)
    para_tmp = {'FF_sup':FF_sup,'SI':SI,'layer':'IG','FF_sup_magn':FF_sup_magn,'unit':unit}
    tmp_df    = pd.DataFrame(para_tmp, index=[indx])
    quencher_DF = quencher_DF.append(tmp_df,sort=True)
    indx += 1

quencher_DF.to_csv('quencher_DF.csv')