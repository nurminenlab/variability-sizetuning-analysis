import sys
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append('C:/Users/lonurmin/Desktop/code/DataAnalysis/')
import data_analysislib as dalib
from matplotlib import gridspec
from statsmodels.formula.api import ols
import scipy as sc
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping


plot_rasters = False
cont_wndw_length = 100
boot_num = int(1e3)

plotter = 'contrast'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
w_dir   = os.getcwd()
anal_type = 'MU'

SUdatfile = 'correlationData_selectedUnits_lenient_macaque-MUA-July2020.pkl'

shitty_fits = [666]
excluded_fits = [666]
                
with open(F_dir + SUdatfile,'rb') as f:
    data = pkl.load(f)

def select_data(spkC, baseline):
    spkC_mean = np.mean(spkC,axis=1)
    baseline_mean = np.mean(baseline)
    
    return (np.max(spkC_mean) - baseline_mean) > 3

def cost_fano(params,xdata,ydata):
    Rhat = dalib.doubleROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

# unit loop
contrast = [100.0]
SI_crit = 0.05
bins = np.arange(-100,600,1)
peani = 'pylly'
virgin = True

# this dataframe holds params for each unit pair
params_df = pd.DataFrame(columns=['maxResponse_u1',
                                  'SI_u1',
                                  'baseline_u1',
                                  'layer_u1',
                                  'SNR_u1',
                                  'maxResponse_u2',
                                  'baseline_u2',
                                  'layer_u2',
                                  'u1_fit_correlation_SML',
                                  'u1_fit_correlation_RF',
                                  'u1_fit_correlation_SUR',
                                  'u1_fit_correlation_LAR',
                                  'u1_fit_correlation_BSL',
                                  'u1_fit_correlation_MIN',
                                  'u1_fit_correlation_MAX',
                                  'u1_fit_correlation_MAX_diam',
                                  'u1_fit_correlation_MIN_diam',
                                  'u2_fit_correlation_SML',
                                  'u2_fit_correlation_RF',
                                  'u2_fit_correlation_SUR',
                                  'u2_fit_correlation_LAR',
                                  'u2_fit_correlation_BSL',
                                  'u2_fit_correlation_MIN',
                                  'u2_fit_correlation_MAX',
                                  'u2_fit_correlation_MAX_diam',
                                  'u2_fit_correlation_MIN_diam',
                                  'gm_fit_correlation_SML',
                                  'gm_fit_correlation_RF',
                                  'gm_fit_correlation_SUR',
                                  'gm_fit_correlation_LAR',
                                  'gm_fit_correlation_BSL',
                                  'gm_fit_correlation_MIN',
                                  'gm_fit_correlation_MAX',
                                  'gm_fit_correlation_MAX_diam',
                                  'gm_fit_correlation_MIN_diam',
                                  'gm_fit_response_SML',
                                  'gm_fit_response_RF',
                                  'gm_fit_response_SUR',
                                  'gm_fit_response_LAR',
                                  'gm_fit_response_atcMAX',
                                  'gm_fit_response_atcMIN',
                                  'ntrials',
                                  'anipe',
                                  'layer_type',
                                  'gm_fit_RF',
                                  'gm_fit_surr',
                                  'u1_fit_RF',
                                  'u1_fit_surr',
                                  'u2_fit_RF',
                                  'u2_fit_surr',
                                  'pair_num'])

correlations = {}
correlations_SGSG = {}
correlations_IGIG = {}
correlations_GG = {}
correlations_all = {}

means = {}
means_SGSG = {}
means_IGIG = {}
means_GG = {}
means_all = {}

indx = 0

# analysis done between these timepoints
anal_duration = 300
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration
count_window = np.array([100])

print('N pairs total ',len(data))
for pair in range(len(data)):

    for cont in contrast:
        if cont in data[pair].keys():
            # data selection
            Y = np.mean(data[pair][cont]['spkC_NoL_pair1'].T,axis=1)
            u1_SI = (np.max(Y) - Y[-1]) / np.max(Y)
            Y = np.mean(data[pair][cont]['spkC_NoL_pair2'].T,axis=1)
            u2_SI = (np.max(Y) - Y[-1]) / np.max(Y)
            
            u1_baseline = np.mean(np.sum(data[pair][cont]['spkR_NoL_pair1'][:,:,bsl_begin:bsl_end],axis=2))
            u2_baseline = np.mean(np.sum(data[pair][cont]['spkR_NoL_pair2'][:,:,bsl_begin:bsl_end],axis=2))
            u1_responsive = select_data(data[pair][cont]['spkC_NoL_pair1'].T,u1_baseline)
            u2_responsive = select_data(data[pair][cont]['spkC_NoL_pair2'].T,u2_baseline)
            gm_baseline = np.sqrt(u1_baseline * u2_baseline)

            anipe = data[pair]['info']['anipe1']
            if u1_responsive and u2_responsive and anipe != 'MM385P1' and anipe != 'MM385P2': # because of the poor quality of the data in MM385P1 and MM385P2
                print('Now analyzing pair ',pair)

                u1_response_mean = np.mean(np.sum(data[pair][100.0]['spkR_NoL_pair1'][:,:,first_tp:last_tp],axis=2),axis=0)
                u1_response_SE = np.std(np.sum(data[pair][100.0]['spkR_NoL_pair1'][:,:,first_tp:last_tp],axis=2),axis=0) / np.sqrt(data[pair][100.0]['spkR_NoL_pair1'].shape[0])
                u2_response_mean = np.mean(np.sum(data[pair][100.0]['spkR_NoL_pair2'][:,:,first_tp:last_tp],axis=2),axis=0)
                u2_response_SE = np.std(np.sum(data[pair][100.0]['spkC_NoL_pair2'][:,:,first_tp:last_tp],axis=2),axis=0) / np.sqrt(data[pair][100.0]['spkR_NoL_pair2'].shape[0])
                
                R1 = np.reshape(u1_response_mean,(1,u1_response_mean.shape[0]))
                R2 = np.reshape(u2_response_mean,(1,u2_response_mean.shape[0]))
                gm_response = sts.mstats.gmean(np.concatenate((R1,R2)))
                gm_response_SE = np.ones(data[pair][100.0]['spkR_NoL_pair1'].shape)
                for pp in range(data[pair][100.0]['spkR_NoL_pair1'].shape[0]):
                    gm_response_SE[pp,:] = np.sqrt(data[pair][100.0]['spkR_NoL_pair1'][pp,:] * data[pair][100.0]['spkR_NoL_pair2'][pp,:])
                    
                # 
                gm_response_SE = np.std(gm_response_SE,axis=0) / np.sqrt(data[pair][100.0]['spkC_NoL_pair1'].shape[0])
                Rsig = np.corrcoef(u1_response_mean, u2_response_mean)[0,1]
                bsl_corr_NoL,bsl_corr_zentral_NoL,cov_NoL = dalib.correlation_binwidths(data[pair][cont]['spkR_NoL_pair1'][:,0,bsl_begin:bsl_end],
                                                                                        data[pair][cont]['spkR_NoL_pair2'][:,0,bsl_begin:bsl_end],
                                                                                        count_window,boot_errs=True,tp1=0,tp2=anal_duration)
                bsl_container = np.mean(bsl_corr_NoL)
                bsl_container_zentral = np.mean(bsl_corr_zentral_NoL)
                
                # loop diams for later averaging
                for stim_diam in range(data[pair][cont]['spkR_NoL_pair1'].shape[1]):
                    rSC = data[pair][100.0]['corr_bin_NoL'][stim_diam,0][0]

                    
                    # grand-average
                    if data[pair]['info']['diam'][stim_diam] in correlations_all.keys():
                        correlations_all[data[pair]['info']['diam'][stim_diam]] = np.concatenate((correlations_all[data[pair]['info']['diam'][stim_diam]],
                                                                                                  np.reshape(rSC,(1,1))), axis=0)
                        means_all[data[pair]['info']['diam'][stim_diam]] = np.concatenate((means_all[data[pair]['info']['diam'][stim_diam]],
                                                                                           np.reshape(gm_response[stim_diam],(1,1))), axis=0)
                    else:
                        correlations_all[data[pair]['info']['diam'][stim_diam]] = np.reshape(rSC,(1,1))
                        means_all[data[pair]['info']['diam'][stim_diam]] = np.reshape(gm_response[stim_diam],(1,1))

                    # group based on layer
                    if data[pair]['info']['L1'] == 'LSG' and data[pair]['info']['L2'] == 'LSG':
                        layer_type = 'SGSG'
                        if data[pair]['info']['diam'][stim_diam] in correlations_SGSG.keys():
                            correlations_SGSG[data[pair]['info']['diam'][stim_diam]] = np.concatenate((correlations_SGSG[data[pair]['info']['diam'][stim_diam]],
                                                                                                       np.reshape(rSC,(1,1))), axis=0)
                            means_SGSG[data[pair]['info']['diam'][stim_diam]] = np.concatenate((means_SGSG[data[pair]['info']['diam'][stim_diam]],
                                                                                                np.reshape(gm_response[stim_diam],(1,1))), axis=0)
                        else:
                            correlations_SGSG[data[pair]['info']['diam'][stim_diam]] = np.reshape(rSC,(1,1))
                            means_SGSG[data[pair]['info']['diam'][stim_diam]] = np.reshape(gm_response[stim_diam],(1,1))
                            
                    elif data[pair]['info']['L1'] == 'L4C' and data[pair]['info']['L2'] == 'L4C':
                        layer_type = 'GG'
                        if data[pair]['info']['diam'][stim_diam] in correlations_GG.keys():
                            correlations_GG[data[pair]['info']['diam'][stim_diam]] = np.concatenate((correlations_GG[data[pair]['info']['diam'][stim_diam]],
                                                                                                     np.reshape(rSC,(1,1))), axis=0)
                            means_GG[data[pair]['info']['diam'][stim_diam]] = np.concatenate((means_GG[data[pair]['info']['diam'][stim_diam]],
                                                                                              np.reshape(gm_response[stim_diam],(1,1))), axis=0)
                        else:
                            correlations_GG[data[pair]['info']['diam'][stim_diam]] = np.reshape(rSC,(1,1))
                            means_GG[data[pair]['info']['diam'][stim_diam]] = np.reshape(gm_response[stim_diam],(1,1))
                            
                    elif data[pair]['info']['L1'] == 'LIG' and data[pair]['info']['L2'] == 'LIG':
                        layer_type = 'IGIG'
                        if data[pair]['info']['diam'][stim_diam] in correlations_IGIG.keys():
                            correlations_IGIG[data[pair]['info']['diam'][stim_diam]] = np.concatenate((correlations_IGIG[data[pair]['info']['diam'][stim_diam]],
                                                                                                       np.reshape(rSC,(1,1))), axis=0)
                            means_IGIG[data[pair]['info']['diam'][stim_diam]] = np.concatenate((means_IGIG[data[pair]['info']['diam'][stim_diam]],
                                                                                                np.reshape(gm_response[stim_diam],(1,1))), axis=0)
                        else:
                            correlations_IGIG[data[pair]['info']['diam'][stim_diam]] = np.reshape(rSC,(1,1))
                            means_IGIG[data[pair]['info']['diam'][stim_diam]] = np.reshape(gm_response[stim_diam],(1,1))
                    else:
                        layer_type = 'mixed'
                        if data[pair]['info']['diam'][stim_diam] in correlations.keys():
                            correlations[data[pair]['info']['diam'][stim_diam]] = np.concatenate((correlations[data[pair]['info']['diam'][stim_diam]],
                                                                                                  np.reshape(rSC,(1,1))), axis=0)
                            means[data[pair]['info']['diam'][stim_diam]] = np.concatenate((means[data[pair]['info']['diam'][stim_diam]],
                                                                                           np.reshape(gm_response[stim_diam],(1,1))), axis=0)
                        else:
                            correlations[data[pair]['info']['diam'][stim_diam]] = np.reshape(rSC,(1,1))
                            means[data[pair]['info']['diam'][stim_diam]] = np.reshape(gm_response[stim_diam],(1,1))
                            
                ##########################
                c_corr = 'go'
                c_bsl  = 'g--'
                wave_c = 'green'
                c_corr_model = 'g-'

                plt.figure(1)
                # ROG fit spike-count data 
                try:
                    u1_popt,pcov = curve_fit(dalib.ROG,data[pair]['info']['diam'],u1_response_mean,bounds=(0,np.inf),maxfev=100000)
                except:
                    args = (data[pair]['info']['diam'],u1_response_mean)
                    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
                    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
                    u1_popt = res.x

                try:
                    u2_popt,pcov = curve_fit(dalib.ROG,data[pair]['info']['diam'],u2_response_mean,bounds=(0,np.inf),maxfev=100000)
                except:
                    args = (data[pair]['info']['diam'],u2_response_mean)
                    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
                    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
                    u2_popt = res.x

                try:
                    gm_popt,pcov = curve_fit(dalib.ROG,data[pair]['info']['diam'],gm_response_mean,bounds=(0,np.inf),maxfev=100000)
                except:
                    args = (data[pair]['info']['diam'],gm_response)
                    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
                    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
                    gm_popt = res.x
                    
                diams_tight = np.logspace(np.log10(data[pair]['info']['diam'][0]),np.log10(data[pair]['info']['diam'][-1]),1000)
                u1_Rhat = dalib.ROG(diams_tight,*u1_popt)
                u2_Rhat = dalib.ROG(diams_tight,*u2_popt)
                gm_Rhat = dalib.ROG(diams_tight,*gm_popt)
                
                # compute gradient for surround size detection
                # u1
                GG = np.gradient(u1_Rhat,diams_tight)
                GG_min_ind = np.argmin(GG)
                try:
                    u1_surr_ind_narrow_new = np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0][0] + GG_min_ind
                    u1_surr_narrow_new  = diams_tight[u1_surr_ind_narrow_new]
                except:
                    u1_surr_ind_narrow_new = -1
                    u1_surr_narrow_new  = diams_tight[u1_surr_ind_narrow_new]

                # u2
                GG = np.gradient(u2_Rhat,diams_tight)
                GG_min_ind = np.argmin(GG)
                try:
                    u2_surr_ind_narrow_new = np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0][0] + GG_min_ind
                    u2_surr_narrow_new  = diams_tight[u2_surr_ind_narrow_new]
                except:
                    u2_surr_ind_narrow_new = -1
                    u2_surr_narrow_new  = diams_tight[u2_surr_ind_narrow_new]
                    
                # gm
                GG = np.gradient(gm_Rhat,diams_tight)
                GG_min_ind = np.argmin(GG)
                try:
                    gm_surr_ind_narrow_new = np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0][0] + GG_min_ind
                    gm_surr_narrow_new  = diams_tight[gm_surr_ind_narrow_new]
                except:
                    gm_surr_ind_narrow_new = -1
                    gm_surr_narrow_new  = diams_tight[gm_surr_ind_narrow_new]
                    
                # ROG fit correlation data 
                a0 = data[pair][100.0]['corr_bin_NoL'].shape[0]
                a1 = data[pair][100.0]['corr_bin_NoL'].shape[1]
                a2 = data[pair][100.0]['corr_bin_NoL_booted'].shape[1]
                
                corrs_mean = np.reshape(data[pair][100.0]['corr_bin_NoL'],(a0,a1))
                corrs_err = np.mean(np.percentile(np.reshape(data[pair][100.0]['corr_bin_NoL_booted'],(a0,a2)),[16,85],axis=1).T,axis=1)
                args = (data[pair]['info']['diam'],corrs_mean[:,0])
                
                bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,-1,-1],[1,30,30,30,100,100,100,100,1,1]]).T
                res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
                Corr_hat = dalib.doubleROG(diams_tight,*res.x)


                # collect parameters from the fits and set bg to gray for shitty fits
                if pair in excluded_fits:
                    u1_fit_correlation_SML = np.nan
                    u1_fit_correlation_RF  = np.nan
                    u1_fit_correlation_SUR = np.nan
                    u1_fit_correlation_LAR = np.nan
                    u1_fit_correlation_BSL = np.nan
                    u1_fit_correlation_MIN = np.nan
                    u1_fit_correlation_MAX = np.nan
                    u1_fit_correlation_MAX_diam = np.nan
                    u1_fit_correlation_MIN_diam = np.nan
                    u1_fit_RF            = np.nan
                    u1_fit_surr          = np.nan

                    u2_fit_correlation_SML = np.nan
                    u2_fit_correlation_RF  = np.nan
                    u2_fit_correlation_SUR = np.nan
                    u2_fit_correlation_LAR = np.nan
                    u2_fit_correlation_BSL = np.nan
                    u2_fit_correlation_MIN = np.nan
                    u2_fit_correlation_MAX = np.nan
                    u2_fit_correlation_MAX_diam = np.nan
                    u2_fit_correlation_MIN_diam = np.nan
                    u2_fit_RF            = np.nan
                    u2_fit_surr          = np.nan

                    gm_fit_correlation_SML = np.nan
                    gm_fit_correlation_RF  = np.nan
                    gm_fit_correlation_SUR = np.nan
                    gm_fit_correlation_LAR = np.nan
                    gm_fit_correlation_BSL = np.nan
                    gm_fit_correlation_MIN = np.nan
                    gm_fit_correlation_MAX = np.nan
                    gm_fit_correlation_MAX_diam = np.nan
                    gm_fit_correlation_MIN_diam = np.nan
                    gm_fit_RF            = np.nan
                    gm_fit_surr          = np.nan

                    gm_fit_response_SML = np.nan
                    gm_fit_response_RF = np.nan
                    gm_fit_response_SUR = np.nan
                    gm_fit_response_LAR = np.nan
                    gm_fit_response_BSL = np.nan
                    gm_fit_response_atcMAX = np.nan
                    gm_fit_response_atcMIN = np.nan
                    pair_num            = np.nan
                    
                else:
                    u1_fit_correlation_SML = Corr_hat[0]
                    u1_fit_correlation_RF  = Corr_hat[np.argmax(u1_Rhat)]
                    u1_fit_correlation_SUR = Corr_hat[u1_surr_ind_narrow_new]
                    u1_fit_correlation_LAR = Corr_hat[-1]
                    u1_fit_correlation_BSL = np.mean(bsl_container)
                    u1_fit_correlation_MIN = np.min(Corr_hat)
                    u1_fit_correlation_MAX = np.max(Corr_hat)
                    u1_fit_correlation_MAX_diam = diams_tight[np.argmax(Corr_hat)]
                    u1_fit_correlation_MIN_diam = diams_tight[np.argmin(Corr_hat)]
                    u1_fit_RF            = diams_tight[np.argmax(u1_Rhat)]
                    u1_fit_surr          = diams_tight[u1_surr_ind_narrow_new]

                    u2_fit_correlation_SML = Corr_hat[0]
                    u2_fit_correlation_RF  = Corr_hat[np.argmax(u2_Rhat)]
                    u2_fit_correlation_SUR = Corr_hat[u2_surr_ind_narrow_new]
                    u2_fit_correlation_LAR = Corr_hat[-1]
                    u2_fit_correlation_BSL = np.mean(bsl_container)
                    u2_fit_correlation_MIN = np.min(Corr_hat)
                    u2_fit_correlation_MAX = np.max(Corr_hat)
                    u2_fit_correlation_MAX_diam = diams_tight[np.argmax(Corr_hat)]
                    u2_fit_correlation_MIN_diam = diams_tight[np.argmin(Corr_hat)]
                    u2_fit_RF            = diams_tight[np.argmax(u2_Rhat)]
                    u2_fit_surr          = diams_tight[u2_surr_ind_narrow_new]

                    gm_fit_correlation_SML = Corr_hat[0]
                    gm_fit_correlation_RF  = Corr_hat[np.argmax(gm_Rhat)]
                    gm_fit_correlation_SUR = Corr_hat[gm_surr_ind_narrow_new]
                    gm_fit_correlation_LAR = Corr_hat[-1]
                    gm_fit_correlation_BSL = np.mean(bsl_container)
                    gm_fit_correlation_MIN = np.min(Corr_hat)
                    gm_fit_correlation_MAX = np.max(Corr_hat)
                    gm_fit_correlation_MAX_diam = diams_tight[np.argmax(Corr_hat)]
                    gm_fit_correlation_MIN_diam = diams_tight[np.argmin(Corr_hat)]
                    gm_fit_RF            = diams_tight[np.argmax(gm_Rhat)]
                    gm_fit_surr          = diams_tight[gm_surr_ind_narrow_new]

                    gm_fit_response_SML = dalib.ROG(0.2,*gm_popt)
                    gm_fit_response_RF = dalib.ROG(diams_tight[np.argmax(gm_Rhat)],*gm_popt)
                    gm_fit_response_SUR = dalib.ROG(diams_tight[gm_surr_ind_narrow_new],*gm_popt)
                    gm_fit_response_LAR = dalib.ROG(25,*gm_popt)
                    gm_fit_response_atcMAX = dalib.ROG(diams_tight[np.argmax(Corr_hat)],*gm_popt)
                    gm_fit_response_atcMIN = dalib.ROG(diams_tight[np.argmin(Corr_hat)],*gm_popt)
                    pair_num            = pair
                
 
                # place pair parameters to a dataframe for later analysis
                para_tmp = {'maxResponse_u1':np.max(u1_response_mean),
                            'SI_u1':u1_SI,
                            'baseline_u1':u1_baseline,
                            'layer_u1':data[pair]['info']['L1'],
                            'maxResponse_u2':np.max(u2_response_mean),
                            'SI_u2':u2_SI,
                            'baseline_u2':u2_baseline,
                            'layer_u2':data[pair]['info']['L2'],
                            'u1_fit_correlation_SML':u1_fit_correlation_SML,
                            'u1_fit_correlation_RF':u1_fit_correlation_RF,
                            'u1_fit_correlation_SUR':u1_fit_correlation_SUR,
                            'u1_fit_correlation_LAR':u1_fit_correlation_LAR,
                            'u1_fit_correlation_BSL':u1_fit_correlation_BSL,
                            'u1_fit_correlation_MIN':u1_fit_correlation_MIN,
                            'u1_fit_correlation_MAX':u1_fit_correlation_MAX,
                            'u1_fit_correlation_MAX_diam':u1_fit_correlation_MAX_diam,
                            'u1_fit_correlation_MIN_diam':u1_fit_correlation_MIN_diam,
                            'u2_fit_correlation_SML':u2_fit_correlation_SML,
                            'u2_fit_correlation_RF':u2_fit_correlation_RF,
                            'u2_fit_correlation_SUR':u2_fit_correlation_SUR,
                            'u2_fit_correlation_LAR':u2_fit_correlation_LAR,
                            'u2_fit_correlation_BSL':u2_fit_correlation_BSL,
                            'u2_fit_correlation_MIN':u2_fit_correlation_MIN,
                            'u2_fit_correlation_MAX':u2_fit_correlation_MAX,
                            'u2_fit_correlation_MAX_diam':u2_fit_correlation_MAX_diam,
                            'u2_fit_correlation_MIN_diam':u2_fit_correlation_MIN_diam,
                            'gm_fit_correlation_SML':gm_fit_correlation_SML,
                            'gm_fit_correlation_RF':gm_fit_correlation_RF,
                            'gm_fit_correlation_SUR':gm_fit_correlation_SUR,
                            'gm_fit_correlation_LAR':gm_fit_correlation_LAR,
                            'gm_fit_correlation_BSL':gm_fit_correlation_BSL,
                            'gm_fit_correlation_MIN':gm_fit_correlation_MIN,
                            'gm_fit_correlation_MAX':gm_fit_correlation_MAX,
                            'gm_fit_correlation_MAX_diam':gm_fit_correlation_MAX_diam,
                            'gm_fit_correlation_MIN_diam':gm_fit_correlation_MIN_diam,
                            'gm_fit_response_SML':gm_fit_response_SML,
                            'gm_fit_response_RF':gm_fit_response_RF,
                            'gm_fit_response_SUR':gm_fit_response_SUR,
                            'gm_fit_response_LAR':gm_fit_response_LAR,
                            'gm_fit_response_atcMAX':gm_fit_response_atcMAX,
                            'gm_fit_response_atcMIN':gm_fit_response_atcMIN,
                            'ntrials':data[pair][100.0]['spkC_NoL_pair1'].shape[0],
                            'anipe':data[pair]['info']['anipe1'],
                            'layer_type':layer_type,
                            'gm_fit_RF':gm_fit_RF,
                            'gm_fit_surr':gm_fit_surr,
                            'u1_fit_RF':u1_fit_RF,
                            'u1_fit_surr':u1_fit_surr,
                            'u2_fit_RF':u2_fit_RF,
                            'u2_fit_surr':u2_fit_surr,
                            'pair_num':pair}


                tmp_df = pd.DataFrame(para_tmp, index=[indx])
                params_df = params_df.append(tmp_df)
                indx = indx + 1



params_df.to_csv(S_dir+'extracted_correlation_params-October-2022.csv')

# save data means
with open(S_dir + 'means.pkl','wb') as f:
    pkl.dump(means,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'means_SGSG.pkl','wb') as f:
    pkl.dump(means_SGSG,f,pkl.HIGHEST_PROTOCOL) 
with open(S_dir + 'means_IGIG.pkl','wb') as f:
    pkl.dump(means_IGIG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'means_GG.pkl','wb') as f:
    pkl.dump(means_GG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'means_all.pkl','wb') as f:
    pkl.dump(means_all,f,pkl.HIGHEST_PROTOCOL)

with open(S_dir + 'correlations.pkl','wb') as f:
    pkl.dump(correlations,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'correlations_SGSG.pkl','wb') as f:
    pkl.dump(correlations_SGSG,f,pkl.HIGHEST_PROTOCOL) 
with open(S_dir + 'correlations_IGIG.pkl','wb') as f:
    pkl.dump(correlations_IGIG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'correlations_GG.pkl','wb') as f:
    pkl.dump(correlations_GG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'correlations_all.pkl','wb') as f:
    pkl.dump(correlations_all,f,pkl.HIGHEST_PROTOCOL)
