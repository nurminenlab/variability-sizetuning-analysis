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
from scipy.optimize import basinhopping, curve_fit

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
MUdatfile = 'selectedData_macaque_Jun2023.pkl'

# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

eps = 0.0000001

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

with open(S_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)

diams = np.array(list(diams_data.keys()))
del(diams_data)
    
with open(S_dir + 'mean_PSTHs_SG-MK-SU-Jun2023.pkl','rb') as f:
    SG_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_SG-MK-SU-Jun2023.pkl','rb') as f:
    SG_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_G-MK-SU-Jun2023.pkl','rb') as f:
    G_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_G-MK-SU-Jun2023.pkl','rb') as f:
    G_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_IG-MK-SU-Jun2023.pkl','rb') as f:
    IG_mn_data = pkl.load(f)    
with open(S_dir + 'vari_PSTHs_IG-MK-SU-Jun2023.pkl','rb') as f:
    IG_vr_data = pkl.load(f)

# param table
quencher_DF = pd.DataFrame(columns=['FF_sup',
                                    'FF_sup_2RF',
                                    'FF_sup_SUR',
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
        fano[stim] = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
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
            fano_boot[boot_num,stim] = np.mean(vari_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window] / (eps + mean_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window]))


    if mn_mtrx.shape[0] == 18:
        my_diams = diams[1:]
    else:
        my_diams = diams

    # get FF @ RF size
    RFind  = np.argmax(FR)
    RF2ind = np.argmin(np.abs(my_diams-2*my_diams[RFind]))
    
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
    
    # 2RF
    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,RF2ind],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,RF2ind] - u2 + uc

    delta_FF = fano[RFind] - fano[RF2ind]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup_2RF = 'suppresser'
        FF_sup_magn_2RF = (fano[RF2ind] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup_2RF = 'facilitator'
        FF_sup_magn = (fano[RF2ind] - fano[RFind]) / fano[RFind]
    else:
        FF_sup_2RF = 'nonether'
        FF_sup_magn = np.nan


    # ROG fit spike-count data 
    try:
        popt,pcov = curve_fit(my_diams,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (my_diams,FR)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        popt = res.x

    diams_tight = np.logspace(np.log10(0.1),np.log10(26),1000)
    Rhat = dalib.ROG(diams_tight,*popt)

    # compute gradient for surround size detection
    GG = np.gradient(Rhat,diams_tight)
    GG_min_ind = np.argmin(GG)
    if GG_min_ind == Rhat.shape[0] - 1:
        surr_ind_narrow_new = Rhat.shape[0] -1
    else:
        if np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0].size != 0:
            surr_ind_narrow_new = np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0][0] + GG_min_ind
        else:
            surr_ind_narrow_new = -1

    surr_narrow_new = diams_tight[surr_ind_narrow_new]
    RF_SURind = np.argmin(np.abs(my_diams-surr_narrow_new))

    # RF SUR
    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,RF_SURind],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,RF_SURind] - u2 + uc

    delta_FF = fano[RFind] - fano[RF_SURind]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup_SUR = 'suppresser'
        FF_sup_magn_SUR = (fano[RF_SURind] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup_SUR = 'facilitator'
        FF_sup_magn = (fano[RF_SURind] - fano[RFind]) / fano[RFind]
    else:
        FF_sup_SUR = 'nonether'
        FF_sup_magn = np.nan

    SI = (np.max(FR) - FR[-1]) / np.max(FR)
    para_tmp = {'FF_sup':FF_sup,'FF_sup_2RF':FF_sup_2RF,'FF_sup_SUR':FF_sup_SUR,'SI':SI,'layer':'SG','FF_sup_magn':FF_sup_magn,'unit':unit}
    
    tmp_df    = pd.DataFrame(para_tmp, index=[indx])
    quencher_DF = pd.concat([quencher_DF,tmp_df],sort=True)
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
        fano[stim] = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
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
            fano_boot[boot_num,stim] = np.mean(vari_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window] / (eps + mean_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window]))
            

    

    if mn_mtrx.shape[0] == 18:
        my_diams = diams[1:]
    else:
        my_diams = diams

    # get FF @ RF size
    RFind  = np.argmax(FR)
    RF2ind = np.argmin(np.abs(my_diams-2*my_diams[RFind]))
    
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
    
    # 2RF
    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,RF2ind],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,RF2ind] - u2 + uc

    delta_FF = fano[RFind] - fano[RF2ind]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup_2RF = 'suppresser'
        FF_sup_magn_2RF = (fano[RF2ind] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup_2RF = 'facilitator'
        FF_sup_magn = (fano[RF2ind] - fano[RFind]) / fano[RFind]
    else:
        FF_sup_2RF = 'nonether'
        FF_sup_magn = np.nan


    # ROG fit spike-count data 
    try:
        popt,pcov = curve_fit(my_diams,FR,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (my_diams,FR)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        popt = res.x

    diams_tight = np.logspace(np.log10(0.1),np.log10(26),1000)
    Rhat = dalib.ROG(diams_tight,*popt)

    # compute gradient for surround size detection
    GG = np.gradient(Rhat,diams_tight)
    GG_min_ind = np.argmin(GG)
    if GG_min_ind == Rhat.shape[0] - 1:
        surr_ind_narrow_new = Rhat.shape[0] -1
    else:
        if np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0].size != 0:
            surr_ind_narrow_new = np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0][0] + GG_min_ind
        else:
            surr_ind_narrow_new = -1

    surr_narrow_new = diams_tight[surr_ind_narrow_new]
    RF_SURind = np.argmin(np.abs(my_diams-surr_narrow_new))

    # RF SUR
    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,RF_SURind],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,RF_SURind] - u2 + uc

    delta_FF = fano[RFind] - fano[RF_SURind]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup_SUR = 'suppresser'
        FF_sup_magn_SUR = (fano[RF_SURind] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup_SUR = 'facilitator'
        FF_sup_magn = (fano[RF_SURind] - fano[RFind]) / fano[RFind]
    else:
        FF_sup_SUR = 'nonether'
        FF_sup_magn = np.nan

    SI = (np.max(FR) - FR[-1]) / np.max(FR)
    para_tmp = {'FF_sup':FF_sup,'FF_sup_2RF':FF_sup_2RF,'FF_sup_SUR':FF_sup_SUR,'SI':SI,'layer':'G','FF_sup_magn':FF_sup_magn,'unit':unit}    
    tmp_df    = pd.DataFrame(para_tmp, index=[indx])
    quencher_DF = pd.concat([quencher_DF,tmp_df],sort=True)
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
        fano[stim] = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
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
            fano_boot[boot_num,stim] = np.mean(vari_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window] / (eps + mean_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window]))


    # get FF @ RF size
    if mn_mtrx.shape[0] == 18:
        my_diams = diams[1:]
    else:
        my_diams = diams

    RFind  = np.argmax(FR)
    RF2ind = np.argmin(np.abs(my_diams-2*my_diams[RFind]))
    
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
    
    # 2RF
    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,RF2ind],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,RF2ind] - u2 + uc

    delta_FF = fano[RFind] - fano[RF2ind]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup_2RF = 'suppresser'
        FF_sup_magn_2RF = (fano[RF2ind] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup_2RF = 'facilitator'
        FF_sup_magn = (fano[RF2ind] - fano[RFind]) / fano[RFind]
    else:
        FF_sup_2RF = 'nonether'
        FF_sup_magn = np.nan


    # ROG fit spike-count data 
    try:
        popt,pcov = curve_fit(my_diams,FR,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (my_diams,FR)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        popt = res.x

    diams_tight = np.logspace(np.log10(0.1),np.log10(26),1000)
    Rhat = dalib.ROG(diams_tight,*popt)

    # compute gradient for surround size detection
    GG = np.gradient(Rhat,diams_tight)
    GG_min_ind = np.argmin(GG)
    if GG_min_ind == Rhat.shape[0] - 1:
        surr_ind_narrow_new = Rhat.shape[0] -1
    else:
        if np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0].size != 0:
            surr_ind_narrow_new = np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0][0] + GG_min_ind
        else:
            surr_ind_narrow_new = -1

    surr_narrow_new = diams_tight[surr_ind_narrow_new]
    RF_SURind = np.argmin(np.abs(stim-surr_narrow_new))

    # RF SUR
    u1 = np.mean(fano_boot[:,RFind],axis=0)
    u2 = np.mean(fano_boot[:,RF_SURind],axis=0)
    uc = (u1 + u2) / 2

    FF_RF = fano_boot[:,RFind] - u1 + uc
    FF_LR = fano_boot[:,RF_SURind] - u2 + uc

    delta_FF = fano[RFind] - fano[RF_SURind]
    delta_FF_boot = FF_RF - FF_LR
    
    if delta_FF > np.percentile(delta_FF_boot,95):
        FF_sup_SUR = 'suppresser'
        FF_sup_magn_SUR = (fano[RF_SURind] - fano[RFind]) / fano[RFind]
    elif delta_FF < np.percentile(delta_FF_boot,5):
        FF_sup_SUR = 'facilitator'
        FF_sup_magn = (fano[RF_SURind] - fano[RFind]) / fano[RFind]
    else:
        FF_sup_SUR = 'nonether'
        FF_sup_magn = np.nan

    SI = (np.max(FR) - FR[-1]) / np.max(FR)
    para_tmp = {'FF_sup':FF_sup,'FF_sup_2RF':FF_sup_2RF,'FF_sup_SUR':FF_sup_SUR,'SI':SI,'layer':'IG','FF_sup_magn':FF_sup_magn,'unit':unit}    
    tmp_df    = pd.DataFrame(para_tmp, index=[indx])
    quencher_DF = pd.concat([quencher_DF,tmp_df],sort=True)
    indx += 1

quencher_DF.to_csv(S_dir+'quencher_DF_rev2.csv')