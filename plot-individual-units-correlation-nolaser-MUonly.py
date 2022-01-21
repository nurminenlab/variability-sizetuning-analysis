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
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration
count_window = np.array([100])

print('N pairs total ',len(data))
for pair in range(2):#range(len(data)):

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

                u1_response_mean = np.mean(data[pair][100.0]['spkC_NoL_pair1'],axis=0)
                u1_response_SE = np.std(data[pair][100.0]['spkC_NoL_pair1'],axis=0) / np.sqrt(data[pair][100.0]['spkC_NoL_pair1'].shape[0])
                u2_response_mean = np.mean(data[pair][100.0]['spkC_NoL_pair2'],axis=0)
                u2_response_SE = np.std(data[pair][100.0]['spkC_NoL_pair2'],axis=0) / np.sqrt(data[pair][100.0]['spkC_NoL_pair2'].shape[0])
                
                R1 = np.reshape(u1_response_mean,(1,u1_response_mean.shape[0]))
                R2 = np.reshape(u2_response_mean,(1,u2_response_mean.shape[0]))
                gm_response = sts.mstats.gmean(np.concatenate((R1,R2)))
                gm_response_SE = np.ones(data[pair][100.0]['spkC_NoL_pair1'].shape)
                for pp in range(data[pair][100.0]['spkC_NoL_pair1'].shape[0]):
                    gm_response_SE[pp,:] = np.sqrt(data[pair][100.0]['spkC_NoL_pair1'][pp,:] * data[pair][100.0]['spkC_NoL_pair2'][pp,:])
                    
                # 
                gm_response_SE = np.std(gm_response_SE,axis=0) / np.sqrt(data[pair][100.0]['spkC_NoL_pair1'].shape[0])
                Rsig = np.corrcoef(u1_response_mean, u2_response_mean)[0,1]
                bsl_corr_NoL,bsl_corr_zentral_NoL,cov_NoL = dalib.correlation_binwidths(data[pair][cont]['spkR_NoL_pair1'][:,0,bsl_begin:bsl_end],
                                                                                        data[pair][cont]['spkR_NoL_pair2'][:,0,bsl_begin:bsl_end],
                                                                                        count_window,boot_errs=True,tp1=0,tp2=anal_duration)
                bsl_container = np.mean(bsl_corr_NoL)
                bsl_container_zentral = np.mean(bsl_corr_zentral_NoL)
                            
                ##########################
                c_corr = 'go'
                c_bsl  = 'g--'
                wave_c = 'green'
                c_corr_model = 'g-'

                plt.figure(1)
                args = (data[pair]['info']['diam'],gm_response)
                bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
                res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
                gm_popt = res.x
                    
                diams_tight = np.logspace(np.log10(data[pair]['info']['diam'][0]),np.log10(data[pair]['info']['diam'][-1]),1000)
                gm_Rhat = dalib.ROG(diams_tight,*gm_popt)
                
                # compute gradient for surround size detection
                # gm
                GG = np.gradient(gm_Rhat,diams_tight)
                GG_min_ind = np.argmin(GG)
                try:
                    gm_surr_ind_narrow_new = np.where(GG[GG_min_ind:] >= 0.1 * GG[GG_min_ind])[0][0] + GG_min_ind
                    gm_surr_narrow_new  = diams_tight[gm_surr_ind_narrow_new]
                except:
                    gm_surr_ind_narrow_new = -1
                    gm_surr_narrow_new  = diams_tight[gm_surr_ind_narrow_new]
                    
                # Fit correlation data 
                a0 = data[pair][100.0]['corr_bin_NoL'].shape[0]
                a1 = data[pair][100.0]['corr_bin_NoL'].shape[1]
                a2 = data[pair][100.0]['corr_bin_NoL_booted'].shape[1]
                
                corrs_mean = np.reshape(data[pair][100.0]['corr_bin_NoL'],(a0,a1))
                corrs_err = np.mean(np.percentile(np.reshape(data[pair][100.0]['corr_bin_NoL_booted'],(a0,a2)),[16,85],axis=1).T,axis=1)
                args = (data[pair]['info']['diam'],corrs_mean[:,0])
                
                bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,-1,-1],[1,30,30,30,100,100,100,100,1,1]]).T
                res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
                Corr_hat = dalib.doubleROG(diams_tight,*res.x)

                # plot rasters, correlation data and fit, spike-count scatter plots, 
                f, ax = plt.subplots(3,5)
                
                

                # collect parameters from the fits and set bg to gray for shitty fits
                if pair in excluded_fits:
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
