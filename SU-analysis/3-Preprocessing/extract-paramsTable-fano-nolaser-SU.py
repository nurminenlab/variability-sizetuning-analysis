import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats as sts
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# please download this library from www.github.com/nurminenlab/Analysis
sys.path.append('C:/Users/lonurmin/Desktop/code/DataAnalysis/')

#import pdb

import data_analysislib as dalib
import pandas as pd
import scipy as sc
from matplotlib import gridspec
from scipy.optimize import basinhopping, curve_fit
from statsmodels.formula.api import ols

plot_rasters = False
cont_wndw_length = 100
boot_num = int(1e3)

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
# path to where the extracted parameters are stored
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
SUdatfile = 'selectedData_macaque_Jun2023.pkl'

# we were not able to fit these units
excluded_fits = []

with open(F_dir + SUdatfile,'rb') as f:
    data = pkl.load(f)

def select_data(spkC, baseline):
    spkC_mean = np.mean(spkC,axis=1)
    baseline_mean = np.mean(baseline)
    
    return (np.max(spkC_mean) - baseline_mean) > 2

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
SI_crit = 0.0
bins = np.arange(-100,600,1)
virgin = True
eps = 0.0000001

# this dataframe holds params for each unit
params_df = pd.DataFrame(columns=['RFdiam',
                                  'maxResponse',
                                  'SI',
                                  'SI_SUR',
                                  'baseline',
                                  'layer',
                                  'anipe',
                                  'animal',                                  
                                  'fit_fano_SML',
                                  'fit_fano_RF',
                                  'fit_fano_SUR',
                                  'fit_fano_LAR',
                                  'fit_fano_BSL',
                                  'fit_fano_MIN',
                                  'fit_fano_MAX',
                                  'fit_fano_MAX_diam',
                                  'fit_fano_MIN_diam', 
                                  'fit_fano_near_SUR',
                                  'spikeWidth',
                                  'spikeSNR',
                                  'unit'])

mean_PSTHs = {}
vari_PSTHs = {}
mean_PSTHs_SG = {}
vari_PSTHs_SG = {}
mean_PSTHs_G = {}
vari_PSTHs_G = {}
mean_PSTHs_IG = {}
vari_PSTHs_IG = {}

mean_PSTHs_narrow = {}
vari_PSTHs_narrow = {}
mean_PSTHs_broad = {}
vari_PSTHs_broad = {}

indx = 0

# analysis done between these timepoints
anal_time = 400
first_tp  = 450
last_tp   = first_tp + anal_time
bsl_begin = 120
bsl_end   = bsl_begin + anal_time
count_window = 100

for unit in range(len(data)):
    print('Unit: ',unit)
    for cont in contrast:
        if cont in data[unit].keys():
            # data selection
            Y = np.mean(data[unit][cont]['spkC_NoL'].T,axis=1)
            SI = (np.max(Y) - Y[-1]) / np.max(Y)
            if select_data(data[unit][cont]['spkC_NoL'].T,data[unit][cont]['baseline']):
                # no-laser
                response    = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)
                response_SE = np.std(data[unit][cont]['spkC_NoL'].T, axis=1) / np.sqrt(data[unit][cont]['spkC_NoL'].T.shape[1])
                
                # stimulus number at fano-peak
                fano_peak_stim = np.argmax(data[unit][cont]['fano_NoL'])+1

                # bootstrapped fano
                fano_boot = data[unit][cont]['boot_fano_NoL'][0:np.argmax(response)+1,:]


                L = data[unit]['info']['layer'].decode('utf-8')

                fano     = data[unit][cont]['fano_NoL']
                fano_LB = fano - np.percentile(data[unit][cont]['boot_fano_NoL'],16,axis=1)
                fano_UB = np.percentile(data[unit][cont]['boot_fano_NoL'],84,axis=1) - fano
                fano_SE = np.vstack((fano_LB, fano_UB))

                anipe = data[unit]['info']['animal'].decode('utf-8') + data[unit]['info']['penetr'].decode('utf-8')
                animal = data[unit]['info']['animal'].decode('utf-8')
                # perform quick anova to see if tuned
                dd = data[unit][100.0]['spkC_NoL']
                dd2 = np.ones((dd.shape[0] * dd.shape[1],2)) * np.nan
                
                for i in range(dd.shape[1]):
                    dd2[0+(i * dd.shape[0]):dd.shape[0] + (i * dd.shape[0]),0] = dd[:,i]
                    dd2[0+(i * dd.shape[0]):dd.shape[0] + (i * dd.shape[0]),1] = i*np.ones(dd.shape[0])
                                                                             
                df = pd.DataFrame(data=dd2, columns=['FR','D'])
                lm = ols('FR ~ C(D)',data=df).fit()
                table = sm.stats.anova_lm(lm,typ=1)
                tuned = table['PR(>F)']['C(D)'] < 0.05

                resp = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)
                vari = np.var(data[unit][cont]['spkC_NoL'].T, axis=1)

                bsl      = np.mean(data[unit][cont]['baseline'])
                bsl_vari = np.var(data[unit][cont]['baseline'])
                 
                # analyze only the units that are tuned 
                if SI >= SI_crit and tuned and data[unit]['info']['SNR1'] >=2.5:

                    fano_container = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    fano_ERR_container = np.nan * np.ones((2,data[unit][cont]['spkR_NoL'].shape[1]))
                    mean_container    = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    mean_SE           = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    bsl_container     = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    
                    mean_PSTH_allstim = np.nan * np.ones((data[unit][cont]['spkR_NoL'].shape[1],1000))
                    vari_PSTH_allstim = np.nan * np.ones((data[unit][cont]['spkR_NoL'].shape[1],1000))
                    
                    # collect FPSTHs
                    for stim_diam in range(data[unit][cont]['spkR_NoL'].shape[1]):
                        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted, vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim_diam,:],
                                                                                                                 count_window,
                                                                                                                 style='same',
                                                                                                                 return_bootdstrs=True,
                                                                                                                 nboots=1000)
                        # use indexing to get rid of redundancy caused by sliding spike-count window
                        mean_SE[stim_diam] = np.std(np.mean(binned_data[:,first_tp:last_tp][:,0:-1:count_window]/(count_window/1000.0),axis=1)) / np.sqrt(binned_data.shape[0])
                        fano_container[stim_diam] = np.mean(vari_PSTH[first_tp:last_tp][0:-1:count_window] / (eps + mean_PSTH[first_tp:last_tp][0:-1:count_window]))
                        
                        fano_booted  = np.mean(np.divide(vari_PSTH_booted[:,first_tp:last_tp][0:-1:count_window], (eps + mean_PSTH_booted[:,first_tp:last_tp][0:-1:count_window])),axis=1)

                        mean_PSTH_allstim[stim_diam,:] = mean_PSTH
                        vari_PSTH_allstim[stim_diam,:] = vari_PSTH

                        CI = np.percentile(fano_booted,[16,84])
                        fano_ERR_container[0,stim_diam] = fano_container[stim_diam] - CI[0]
                        fano_ERR_container[1,stim_diam] = CI[1] - fano_container[stim_diam]
                        mean_container[stim_diam] = np.mean(binned_data[:,first_tp:last_tp][:,0:-1:count_window]/(count_window/1000.0))
                        bsl_container[stim_diam]  = np.mean(vari_PSTH[bsl_begin:bsl_end][0:-1:count_window] / (eps + mean_PSTH[bsl_begin:bsl_end][0:-1:count_window]))
                        
                        if data[unit]['info']['diam'][stim_diam] in mean_PSTHs.keys():
                            mean_PSTHs[unit] = np.concatenate((mean_PSTHs[data[unit]['info']['diam'][stim_diam]],
                                                                                                np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))), axis=0)
                            vari_PSTHs[unit] = np.concatenate((vari_PSTHs[data[unit]['info']['diam'][stim_diam]],
                                                                                                np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))), axis=0)
                        else:
                            mean_PSTHs[unit] = np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))
                            vari_PSTHs[unit] = np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))

                    # layer resolved data
                    if L == 'LSG':                            
                        mean_PSTHs_SG[unit] = mean_PSTH_allstim
                        vari_PSTHs_SG[unit] = vari_PSTH_allstim

                    elif L == 'L4C':
                        mean_PSTHs_G[unit] = mean_PSTH_allstim
                        vari_PSTHs_G[unit] = vari_PSTH_allstim

                    elif L == 'LIG' :
                        mean_PSTHs_IG[unit] = mean_PSTH_allstim
                        vari_PSTHs_IG[unit] = vari_PSTH_allstim

                    else:
                        print('This should not happen')


                    max_ind = np.argmax(mean_container)
                    
                        
                    if virgin:
                        fano = np.reshape(fano, (1,fano.shape[0]))
                        resp = np.reshape(resp, (1,resp.shape[0]))
                        vari = np.reshape(vari, (1,vari.shape[0]))
                        bsl_all       = np.reshape(bsl,(1,1))
                        bsl_vari_all  = np.reshape(bsl_vari, (1,1))
                        
                        if anipe == 'MK366P3':
                            # because MK366P3 did not have 0.1 diam grating
                            fano_all = np.concatenate((np.nan * np.ones((1,1)), fano))
                            resp_all = np.concatenate((np.nan * np.ones((1,1)), resp))
                            vari_all = np.concatenate((np.nan * np.ones((1,1)), vari))
                        else:
                            fano_all = fano
                            resp_all = resp
                            vari_all = vari
                            
                        virgin = False
                    else:

                        bsl_all       = np.concatenate((bsl_all, np.reshape(bsl,(1,1))), axis=0)
                        bsl_vari_all  = np.concatenate((bsl_vari_all, np.reshape(bsl_vari,(1,1))),axis=0)
                        
                        fano = np.reshape(fano, (1,fano.shape[0]))
                        resp = np.reshape(resp, (1,resp.shape[0]))
                        vari = np.reshape(vari, (1,vari.shape[0]))
                        if anipe == 'MK366P3':
                            fano     = np.concatenate((np.nan * np.ones((1,1)), fano),axis=1)
                            fano_all = np.concatenate((fano_all,fano),axis=0)
                            resp     = np.concatenate((np.nan * np.ones((1,1)), resp),axis=1)
                            resp_all = np.concatenate((resp_all,resp),axis=0)
                            vari     = np.concatenate((np.nan * np.ones((1,1)), vari),axis=1)
                            vari_all = np.concatenate((vari_all,vari),axis=0)
                            
                        else:
                            fano_all = np.concatenate((fano_all,fano),axis=0)
                            resp_all = np.concatenate((resp_all,resp),axis=0)
                            vari_all = np.concatenate((vari_all,vari),axis=0)
                            
                    ##########################
                   

                    # ROG fit spike-count data 
                    """ try:
                        popt,pcov = curve_fit(dalib.ROG,data[unit]['info']['diam'],mean_container,bounds=(0,np.inf),maxfev=100000)
                    except: """
                    args = (data[unit]['info']['diam'],mean_container)
                    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
                    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
                    popt = res.x
                        
                    diams_tight = np.logspace(np.log10(data[unit]['info']['diam'][0]),np.log10(data[unit]['info']['diam'][-1]),1000)
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
                    # ROG fit fano data 
                    args = (data[unit]['info']['diam'],fano_container)
                    
                    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
                    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)

                    Fhat = dalib.doubleROG(diams_tight,*res.x)                   
                    
                    # collect parameters from the fits and set bg to gray for shitty fits
                    if unit in excluded_fits:                        
                        fit_fano_SML = np.nan
                        fit_fano_RF  = np.nan
                        fit_fano_SUR = np.nan
                        fit_fano_LAR = np.nan
                        fit_fano_BSL = np.nan
                        fit_fano_MIN = np.nan
                        fit_fano_MAX = np.nan
                        fit_fano_near_SUR = np.nan

                        fit_fano_MAX_diam = np.nan
                        fit_fano_MIN_diam = np.nan
                        fit_RF            = np.nan
                        fit_surr          = np.nan
                    else:                        
                        fit_fano_SML = Fhat[0]
                        fit_fano_RF  = Fhat[np.argmax(Rhat)]
                        fit_fano_SUR = Fhat[surr_ind_narrow_new]
                        fit_fano_LAR = Fhat[-1]
                        fit_fano_BSL = np.mean(bsl_container)
                        fit_fano_MIN = np.max((np.min(Fhat),0)) # in case of negative values resulting from bad fits
                        fit_fano_MAX = np.max(Fhat)
                        fit_fano_near_SUR = dalib.doubleROG(diams_tight[np.argmax(Rhat)]*2,*res.x)

                        fit_fano_MAX_diam = diams_tight[np.argmax(Fhat)]
                        fit_fano_MIN_diam = diams_tight[np.argmin(Fhat)]
                        fit_RF            = diams_tight[np.argmax(Rhat)]
                        fit_surr          = diams_tight[surr_ind_narrow_new]


                    ##########################
                    surr,surr_ind = dalib.saturation_point(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1),data[unit]['info']['diam'],criteria=1.3)
                    ntrials = data[unit][cont]['spkC_NoL'].shape[0]
                    mx_ind = np.argmax(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1))
                    spkC = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)                    

                    # place unit parameters to a dataframe for later analysis
                    para_tmp = np.ones((1,10),dtype=object)*np.nan
                    para_tmp = {'RFdiam':data[unit]['info']['diam'][mx_ind],
                                'maxResponse':np.max(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)),
                                'SI':(np.max(Rhat) - Rhat[-1]) / np.max(Rhat),
                                'SI_SUR':(np.max(Rhat) - Rhat[surr_ind_narrow_new]) / np.max(Rhat),
                                'baseline':np.mean(data[unit][100.0]['baseline']),
                                'layer':L,
                                'anipe':anipe,
                                'animal':animal,
                                'ntrials':ntrials,                                
                                'fit_fano_SML':fit_fano_SML,
                                'fit_fano_RF':fit_fano_RF,
                                'fit_fano_SUR':fit_fano_SUR,
                                'fit_fano_LAR':fit_fano_LAR,
                                'fit_fano_BSL':fit_fano_BSL,
                                'fit_fano_MIN':fit_fano_MIN,
                                'fit_fano_MAX':fit_fano_MAX,
                                'fit_fano_MIN_diam':fit_fano_MIN_diam,
                                'fit_fano_MAX_diam':fit_fano_MAX_diam,
                                'fit_fano_near_SUR':fit_fano_near_SUR,
                                'spikeWidth':data[unit]['info']['spikewidth1'],
                                'spikeSNR':data[unit]['info']['SNR1'],
                                'unit':unit}

                    tmp_df = pd.DataFrame(para_tmp, index=[indx])
                    params_df = params_df.append(tmp_df,sort=True)
                    indx = indx + 1
                    
month = datetime.now().strftime('%b') 
year = datetime.now().strftime('%Y')

# save data
params_df.to_csv(S_dir+'SU-extracted_params-'+month+year+'.csv')

with open(S_dir + 'mean_PSTHs-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(mean_PSTHs,f,pkl.HIGHEST_PROTOCOL)

with open(S_dir + 'vari_PSTHs-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(vari_PSTHs,f,pkl.HIGHEST_PROTOCOL)

# layer resolved
# SG
with open(S_dir + 'mean_PSTHs_SG-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(mean_PSTHs_SG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'vari_PSTHs_SG-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(vari_PSTHs_SG,f,pkl.HIGHEST_PROTOCOL)
# G
with open(S_dir + 'mean_PSTHs_G-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(mean_PSTHs_G,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'vari_PSTHs_G-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(vari_PSTHs_G,f,pkl.HIGHEST_PROTOCOL)
# IG
with open(S_dir + 'mean_PSTHs_IG-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(mean_PSTHs_IG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'vari_PSTHs_IG-MK-SU-'+month+year+'.pkl','wb') as f:
    pkl.dump(vari_PSTHs_IG,f,pkl.HIGHEST_PROTOCOL)