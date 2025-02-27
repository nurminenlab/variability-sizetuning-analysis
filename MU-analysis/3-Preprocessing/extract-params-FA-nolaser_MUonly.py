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

save_figures = False

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'

data_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/2-PrecomputedAnalysis/'

SG_netvariance = np.load(data_dir+'netvariance_all_SG.npy')
G_netvariance  = np.load(data_dir+'netvariance_all_G.npy')
IG_netvariance = np.load(data_dir+'netvariance_all_IG.npy')

bsl_SG_netvariance = np.load(data_dir+'bsl_netvariance_all_SG.npy')
bsl_G_netvariance  = np.load(data_dir+'bsl_netvariance_all_G.npy')
bsl_IG_netvariance = np.load(data_dir+'bsl_netvariance_all_IG.npy')

SG_meanresponses = np.load(data_dir+'mean_response_all_SG.npy')
G_meanresponses  = np.load(data_dir+'mean_response_all_G.npy')
IG_meanresponses = np.load(data_dir+'mean_response_all_IG.npy')

SG_unit_animals = np.int16(np.load(data_dir+'SG_unit_animals.npy')).flatten()
G_unit_animals  = np.int16(np.load(data_dir+'G_unit_animals.npy')).flatten()
IG_unit_animals = np.int16(np.load(data_dir+'IG_unit_animals.npy')).flatten()

count_window = 100

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

def cost_fano(params,xdata,ydata):
    Rhat = dalib.doubleROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err


# this dataframe holds params for each unit
params_df = pd.DataFrame(columns=['layer',
                                'anipe',
                                'animal',
                                'ntrials',                                  
                                'fit_FA_SML',
                                'fit_FA_RF',
                                'fit_FA_SUR',
                                'fit_FA_SUR_200',
                                'fit_FA_SUR_400',
                                'fit_FA_SUR_800',
                                'fit_FA_LAR',
                                'fit_FA_BSL',
                                'fit_FA_MIN',
                                'fit_FA_MAX',
                                'fit_FA_MAX_diam',
                                'fit_FA_MIN_diam',
                                'sur_MAX_diam',
                                'fit_RF'])


# loop
indx = 0
# SG
#----------------------------------------------------------
for u in range(SG_netvariance.shape[0]):
    print('Now analysing unit ',u)

    if np.isnan(SG_netvariance[u,0]):
        diams = np.array([0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26])
        diams_tight = np.logspace(np.log10(diams[0]),np.log10(diams[-1]),1000)
        FR = SG_meanresponses[u,1:]
        FA = SG_netvariance[u,1:]
    else:
        diams = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26])
        diams_tight = np.logspace(np.log10(diams[0]),np.log10(diams[-1]),1000)
        FR = SG_meanresponses[u,:]
        FA = SG_netvariance[u,:]
    # fit FR data
    try:
        FR_popt,pcov = curve_fit(dalib.ROG,diams,FR,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (diams,FR)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        FR_popt = res.x

    # fit netvariance data
    args = (diams,FA)
    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
    FA_popt = res.x

    Rhat  = dalib.ROG(diams_tight,*FR_popt)
    FAhat = dalib.doubleROG(diams_tight,*FA_popt)
    RF2_i   = 2*diams_tight[np.argmax(Rhat)]
    RF2_i   = np.argmin(np.abs(RF2_i - diams_tight))
    RF4_i   = 4*diams_tight[np.argmax(Rhat)]
    RF4_i   = np.argmin(np.abs(RF4_i - diams_tight))
    RF8_i   = 8*diams_tight[np.argmax(Rhat)]
    RF8_i   = np.argmin(np.abs(RF8_i - diams_tight))
    
    FAhat_temp = FAhat[np.argmax(Rhat):]
    diams_tight_temp = diams_tight[np.argmax(Rhat):]                    #     
    sur_MAX_diam = diams_tight_temp[np.argmax(FAhat_temp)]
    
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
    
    # 
    surr_narrow_new = diams_tight[surr_ind_narrow_new]

    BSL = np.nanmean(bsl_SG_netvariance[u,:])
    para_tmp = {'layer':'SG',
                'animal':'MK'+str(SG_unit_animals[u]),                                                               
                'fit_FA_SML':FAhat[0],
                'fit_FA_RF':FAhat[np.argmax(Rhat)],
                'fit_FA_SUR_200':FAhat[RF2_i],
                'fit_FA_SUR_400':FAhat[RF4_i],
                'fit_FA_SUR_800':FAhat[RF8_i],
                'fit_FA_SUR':FAhat[surr_ind_narrow_new],
                'fit_FA_LAR':FAhat[-1],                
                'fit_FA_MIN':np.max((np.min(FAhat),0)), # in case of negative values resulting from bad fits,
                'fit_FA_MAX':np.max(FAhat),
                'fit_FA_MAX_diam':diams_tight[np.argmax(FAhat)],
                'fit_FA_MIN_diam':diams_tight[np.argmin(FAhat)],
                'fit_FA_BSL':BSL,
                'fit_RF':diams_tight[np.argmax(Rhat)],
                'sur_MAX_diam':sur_MAX_diam}

    tmp_df = pd.DataFrame(para_tmp, index=[indx])
    params_df = params_df.append(tmp_df,sort=True)
    indx = indx + 1


# G
#----------------------------------------------------------
for u in range(G_netvariance.shape[0]):
    if np.isnan(G_netvariance[u,0]):
        diams = np.array([0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26])
        diams_tight = np.logspace(np.log10(diams[0]),np.log10(diams[-1]),1000)
        FR = G_meanresponses[u,1:]
        FA = G_netvariance[u,1:]
    else:
        diams = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26])
        diams_tight = np.logspace(np.log10(diams[0]),np.log10(diams[-1]),1000)
        FR = G_meanresponses[u,:]
        FA = G_netvariance[u,:]

    # fit FR data
    try:
        FR_popt,pcov = curve_fit(dalib.ROG,diams, FR,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (diams,FR)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        FR_popt = res.x

    # fit netvariance data
    args = (diams,FA)
    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
    FA_popt = res.x

    Rhat  = dalib.ROG(diams_tight,*FR_popt)
    FAhat = dalib.doubleROG(diams_tight,*FA_popt)
    RF2_i   = 2*diams_tight[np.argmax(Rhat)]
    RF2_i   = np.argmin(np.abs(RF2_i - diams_tight))
    RF4_i   = 4*diams_tight[np.argmax(Rhat)]
    RF4_i   = np.argmin(np.abs(RF4_i - diams_tight))
    RF8_i   = 8*diams_tight[np.argmax(Rhat)]
    RF8_i   = np.argmin(np.abs(RF8_i - diams_tight))

    FAhat_temp = FAhat[np.argmax(Rhat):]
    diams_tight_temp = diams_tight[np.argmax(Rhat):]                    #     
    sur_MAX_diam = diams_tight_temp[np.argmax(FAhat_temp)]

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
    
    # 
    surr_narrow_new = diams_tight[surr_ind_narrow_new]
    BSL = np.nanmean(bsl_G_netvariance[u,:])
    para_tmp = {'layer':'G',
                'animal':'MK'+str(G_unit_animals[u]),
                'fit_FA_SML':FAhat[0],
                'fit_FA_RF':FAhat[np.argmax(Rhat)],
                'fit_FA_SUR_200':FAhat[RF2_i],
                'fit_FA_SUR_400':FAhat[RF4_i],
                'fit_FA_SUR_800':FAhat[RF8_i],
                'fit_FA_SUR':FAhat[surr_ind_narrow_new],
                'fit_FA_LAR':FAhat[-1],                
                'fit_FA_MIN':np.max((np.min(FAhat),0)), # in case of negative values resulting from bad fits,
                'fit_FA_MAX':np.max(FAhat),
                'fit_FA_MAX_diam':diams_tight[np.argmax(FAhat)],
                'fit_FA_MIN_diam':diams_tight[np.argmin(FAhat)],
                'fit_FA_BSL':BSL,
                'fit_RF':diams_tight[np.argmax(Rhat)],
                'sur_MAX_diam':sur_MAX_diam}

    tmp_df = pd.DataFrame(para_tmp, index=[indx])
    params_df = params_df.append(tmp_df,sort=True)
    indx = indx + 1

# IG
#----------------------------------------------------------
for u in range(IG_netvariance.shape[0]):
    if np.isnan(IG_netvariance[u,0]):
        diams = np.array([0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26])
        diams_tight = np.logspace(np.log10(diams[0]),np.log10(diams[-1]),1000)
        FR = IG_meanresponses[u,1:]
        FA = IG_netvariance[u,1:]
    else:
        diams = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26])
        diams_tight = np.logspace(np.log10(diams[0]),np.log10(diams[-1]),1000)
        FR = IG_meanresponses[u,:]
        FA = IG_netvariance[u,:]

    # fit FR data
    try:
        FR_popt,pcov = curve_fit(dalib.ROG,diams, FR,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (diams,FR)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        FR_popt = res.x

    # fit netvariance data
    args = (diams,FA)
    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
    FA_popt = res.x

    Rhat  = dalib.ROG(diams_tight,*FR_popt)
    FAhat = dalib.doubleROG(diams_tight,*FA_popt)
    RF2_i   = 2*diams_tight[np.argmax(Rhat)]
    RF2_i   = np.argmin(np.abs(RF2_i - diams_tight))
    RF4_i   = 4*diams_tight[np.argmax(Rhat)]
    RF4_i   = np.argmin(np.abs(RF4_i - diams_tight))
    RF8_i   = 8*diams_tight[np.argmax(Rhat)]
    RF8_i   = np.argmin(np.abs(RF8_i - diams_tight))

    FAhat_temp = FAhat[np.argmax(Rhat):]
    diams_tight_temp = diams_tight[np.argmax(Rhat):]                    #     
    sur_MAX_diam = diams_tight_temp[np.argmax(FAhat_temp)]

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
    
    #    
    surr_narrow_new = diams_tight[surr_ind_narrow_new]
    BSL = np.nanmean(bsl_IG_netvariance[u,:])
    para_tmp = {'layer':'IG',
                'animal':'MK'+str(IG_unit_animals[u]),                                                               
                'fit_FA_SML':FAhat[0],
                'fit_FA_RF':FAhat[np.argmax(Rhat)],
                'fit_FA_SUR':FAhat[surr_ind_narrow_new],
                'fit_FA_SUR_200':FAhat[RF2_i],
                'fit_FA_SUR_400':FAhat[RF4_i],
                'fit_FA_SUR_800':FAhat[RF8_i],
                'fit_FA_LAR':FAhat[-1],                
                'fit_FA_MIN':np.max((np.min(FAhat),0)), # in case of negative values resulting from bad fits,
                'fit_FA_MAX':np.max(FAhat),
                'fit_FA_MAX_diam':diams_tight[np.argmax(FAhat)],
                'fit_FA_MIN_diam':diams_tight[np.argmin(FAhat)],
                'fit_FA_BSL':BSL,
                'fit_RF':diams_tight[np.argmax(Rhat)],
                'sur_MAX_diam':sur_MAX_diam}

    tmp_df = pd.DataFrame(para_tmp, index=[indx])
    params_df = params_df.append(tmp_df,sort=True)
    indx = indx + 1

# save data
params_df.to_csv(S_dir+'FA-params-May-2024.csv')