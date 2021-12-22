import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sc
sys.path.append('C:/Users/lonurmin/Desktop/code/DataAnalysis')
import data_analysislib as dalib
from matplotlib import gridspec
from statsmodels.formula.api import ols
import scipy as sc
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import basinhopping
import pdb

plot_rasters = False
cont_wndw_length = 100
boot_num = int(1e3)

plotter = 'contrast'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/localDATA/UTAH/CorrelatedVariability/results/paper_v9/MK-MU/'
w_dir   = os.getcwd()
anal_type = 'SU'

plt.figure(1,figsize=(11.7,8.3))
plt.figure(2,figsize=(11.7,8.3))

examples_pdf = PdfPages(S_dir + 'size-tuning-N-fano-MK-MUA-Dec2021.pdf')
SUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

shitty_fits = [2,5,6,7,13,19,35,58,53,57,59,68,70,71,72,77,79,90,92,98,105]
excluded_fits = [72]

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
virgin = True

# this dataframe holds params for each unit
params_df = pd.DataFrame(columns=['RFdiam','maxResponse','SI','baseline',
                                  'layer','anipe','center_slope','surround_slope','centerSIG',
                                  'center_slope_fano','surround_slope_fano','ntrials','surroundSIG_fano',
                                  'center_slope_narrow_window','surround_slope_narrow_window',
                                  'fit_fano_SML','fit_fano_RF',
                                  'fit_fano_SUR','fit_fano_LAR',
                                  'fit_fano_BSL','fit_fano_MIN',
                                  'fit_fano_MAX','fit_fano_MAX_diam',
                                  'fit_fano_MIN_diam'])
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

                # suppressive region
                fano_tailing     = data[unit][cont]['fano_NoL'][np.argmax(response):]
                fano_tailing_LB  = fano_tailing - np.percentile(data[unit][cont]['boot_fano_NoL'][np.argmax(response):,:],16,axis=1)
                fano_tailing_UB  = np.percentile(data[unit][cont]['boot_fano_NoL'][np.argmax(response):,:],84,axis=1) - fano_tailing
                fano_tailing_SE_boot = np.vstack((fano_tailing_LB, fano_tailing_UB))
                
                response_SE_tailing = response_SE[np.argmax(response):]
                response_tailing = response[np.argmax(response):]

                response = response[0:np.argmax(response)+1]

                L = data[unit]['info']['layer'].decode('utf-8')

                fano     = data[unit][cont]['fano_NoL']
                fano_LB = fano - np.percentile(data[unit][cont]['boot_fano_NoL'],16,axis=1)
                fano_UB = np.percentile(data[unit][cont]['boot_fano_NoL'],84,axis=1) - fano
                fano_SE = np.vstack((fano_LB, fano_UB))

                anipe = data[unit]['info']['animal'].decode('utf-8') + data[unit]['info']['penetr'].decode('utf-8')

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
                # remove MM385 because the data really looks like there something wrong
                if fano_tailing.shape[0] > 1 and SI >= SI_crit and anipe != 'MM385P1' and anipe != 'MM385P2' and tuned:

                    fano_container = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    fano_ERR_container = np.nan * np.ones((2,data[unit][cont]['spkR_NoL'].shape[1]))
                    mean_container    = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    mean_SE           = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    bsl_container     = np.nan * np.ones(data[unit][cont]['spkR_NoL'].shape[1])
                    
                    # collect FPSTHs
                    for stim_diam in range(data[unit][cont]['spkR_NoL'].shape[1]):
                        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted, vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim_diam,:],
                                                                                                                 count_window,style='same',return_bootdstrs=True,nboots=1000)
                        # use indexing to get rid of redundancy caused by sliding spike-count window
                        mean_SE[stim_diam] = np.std(np.mean(binned_data[:,first_tp:last_tp][:,0:-1:count_window]/(count_window/1000.0),axis=1)) / np.sqrt(binned_data.shape[0])
                        fano_results = sm.OLS(vari_PSTH[first_tp:last_tp][0:-1:count_window],mean_PSTH[first_tp:last_tp][0:-1:count_window]).fit()
                        fano_booted  = np.nan * np.ones(mean_PSTH_booted.shape[0])
                        # compute bootstrapped fano time-course
                        for boot_num in range(mean_PSTH_booted.shape[0]):
                            fano_booted[boot_num] = sm.OLS(vari_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window],
                                                           mean_PSTH_booted[boot_num,first_tp:last_tp][0:-1:count_window]).fit().params[0]
                            
                        
                        fano_container[stim_diam] = fano_results.params[0]
                        pdb.set_trace()
                        CI = np.percentile(fano_booted,[16,84])
                        fano_ERR_container[0,stim_diam] = fano_container[stim_diam] - CI[0]
                        fano_ERR_container[1,stim_diam] = CI[1] - fano_container[stim_diam]
                        mean_container[stim_diam] = np.mean(binned_data[:,first_tp:last_tp][:,0:-1:count_window]/(count_window/1000.0))
                        bsl_container[stim_diam] = np.mean(mean_PSTH[bsl_begin:bsl_end])
                        
                        
                        if data[unit]['info']['diam'][stim_diam] in mean_PSTHs.keys():
                            mean_PSTHs[data[unit]['info']['diam'][stim_diam]] = np.concatenate((mean_PSTHs[data[unit]['info']['diam'][stim_diam]],
                                                                                                np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))), axis=0)
                            vari_PSTHs[data[unit]['info']['diam'][stim_diam]] = np.concatenate((vari_PSTHs[data[unit]['info']['diam'][stim_diam]],
                                                                                                np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))), axis=0)
                        else:
                            mean_PSTHs[data[unit]['info']['diam'][stim_diam]] = np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))
                            vari_PSTHs[data[unit]['info']['diam'][stim_diam]] = np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))

                        # layer resolved data
                        if L == 'LSG' and data[unit]['info']['diam'][stim_diam] in mean_PSTHs_SG.keys():
                            pdb.set_trace()
                            mean_PSTHs_SG[data[unit]['info']['diam'][stim_diam]] = np.concatenate((mean_PSTHs_SG[data[unit]['info']['diam'][stim_diam]],
                                                                                                   np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))), axis=0)
                            vari_PSTHs_SG[data[unit]['info']['diam'][stim_diam]] = np.concatenate((vari_PSTHs_SG[data[unit]['info']['diam'][stim_diam]],
                                                                                                   np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))), axis=0)

                        elif L == 'LSG' and data[unit]['info']['diam'][stim_diam] not in mean_PSTHs_SG.keys():
                            mean_PSTHs_SG[data[unit]['info']['diam'][stim_diam]] = np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))
                            vari_PSTHs_SG[data[unit]['info']['diam'][stim_diam]] = np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))

                        elif L == 'LIG' and data[unit]['info']['diam'][stim_diam] in mean_PSTHs_IG.keys():
                            mean_PSTHs_IG[data[unit]['info']['diam'][stim_diam]] = np.concatenate((mean_PSTHs_IG[data[unit]['info']['diam'][stim_diam]],
                                                                                                   np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))), axis=0)
                            vari_PSTHs_IG[data[unit]['info']['diam'][stim_diam]] = np.concatenate((vari_PSTHs_IG[data[unit]['info']['diam'][stim_diam]],
                                                                                                   np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))), axis=0)

                        elif L == 'LIG' and data[unit]['info']['diam'][stim_diam] not in mean_PSTHs_IG.keys():
                            mean_PSTHs_IG[data[unit]['info']['diam'][stim_diam]] = np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))
                            vari_PSTHs_IG[data[unit]['info']['diam'][stim_diam]] = np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))

                        elif L == 'L4C' and data[unit]['info']['diam'][stim_diam] in mean_PSTHs_G.keys():
                            mean_PSTHs_G[data[unit]['info']['diam'][stim_diam]] = np.concatenate((mean_PSTHs_G[data[unit]['info']['diam'][stim_diam]],
                                                                                                  np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))), axis=0)
                            vari_PSTHs_G[data[unit]['info']['diam'][stim_diam]] = np.concatenate((vari_PSTHs_G[data[unit]['info']['diam'][stim_diam]],
                                                                                                  np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))), axis=0)

                        elif L == 'L4C' and data[unit]['info']['diam'][stim_diam] not in mean_PSTHs_G.keys():
                            mean_PSTHs_G[data[unit]['info']['diam'][stim_diam]] = np.reshape(mean_PSTH,(1,mean_PSTH.shape[0]))
                            vari_PSTHs_G[data[unit]['info']['diam'][stim_diam]] = np.reshape(vari_PSTH,(1,vari_PSTH.shape[0]))

                        else:
                            print('This should not happen')

                        
                    max_ind = np.argmax(mean_container)
                    if max_ind == 0:
                        center_slope_narrow_window = np.nan
                    else:
                        X = sm.add_constant(mean_container[0:max_ind+1])
                        center_slope_narrow_window = sm.OLS(fano_container[0:max_ind+1], X).fit().params[1]

                    if max_ind == data[unit]['info']['diam'].shape[0]-1:
                        surround_slope_narrow_window = np.nan
                    else:
                        X = sm.add_constant(mean_container[max_ind:])
                        surround_slope_narrow_window = sm.OLS(fano_container[max_ind:], X).fit().params[1]
                        
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
                    f_fano = 'ro'
                    f_bsl  = 'r--'
                    wave_c = 'red'
                    f_fano_model = 'r-'
                        

                    plt.figure(1)
                    # 1st row
                    # fano-vs-size
                    surr,surr_ind = dalib.saturation_point(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1),data[unit]['info']['diam'],criteria=1.3)
                    ax1 = plt.subplot(4,4,1)
                    ax1b = ax1.twinx()
                    ntrials = data[unit][cont]['spkC_NoL'].shape[0]
                    ax1.errorbar(data[unit]['info']['diam'], np.mean(data[unit][cont]['spkC_NoL'].T, axis=1),
                                 yerr=np.std(data[unit][cont]['spkC_NoL'].T, axis=1) / np.sqrt(data[unit][cont]['spkC_NoL'].T.shape[1]), fmt='ko-',label='spike-count')
                    ax1.plot([data[unit]['info']['diam'][0],data[unit]['info']['diam'][-1]],[np.mean(data[unit][cont]['baseline']), np.mean(data[unit][cont]['baseline'])],'ko--')
                    ax1.plot([surr,surr],[np.mean(data[unit][cont]['baseline']), np.max(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1))],'g--')
                    ax1.set_xscale('log')
                    ax1.set_xlabel('Diameter (deg)')
                    ax1.set_ylabel('Spike-count')
                    ax1b.set_xscale('log')
                    if np.isnan(fano[0,0]):
                        fano = fano[0,1:]
                    ax1b.errorbar(data[unit]['info']['diam'], np.squeeze(fano), yerr=fano_SE, fmt=f_fano,label='fano-factor')
                    ax1b.plot([data[unit]['info']['diam'][0],data[unit]['info']['diam'][-1]],[np.mean(data[unit][cont]['fano_bsl']), np.mean(data[unit][cont]['fano_bsl'])],f_bsl)
                    ax1b.set_ylabel('Fano-factor')
                    plt.legend()
                    plt.title('unit '+ str(unit) + anipe + ' layer ' + L)
                    
                    # fano-vs-size for small count window
                    surr_narrow,surr_ind_narrow = dalib.saturation_point(mean_container,data[unit]['info']['diam'],criteria=1.3)

                    # ROG fit spike-count data 
                    try:
                        popt,pcov = curve_fit(dalib.ROG,data[unit]['info']['diam'],mean_container,bounds=(0,np.inf),maxfev=100000)
                    except:
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
                    if unit in shitty_fits:
                        bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
                        res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
                    else:
                        res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args},seed=1234,niter=1000)

                    Fhat = dalib.doubleROG(diams_tight,*res.x)

                    ax2 = plt.subplot(4,4,2)
                    ax2b = ax2.twinx() 
                    ax2.errorbar(data[unit]['info']['diam'],mean_container,yerr=mean_SE,fmt='ko',label='spike-count')
                    ax2.plot([data[unit]['info']['diam'][0],data[unit]['info']['diam'][-1]],[np.mean(data[unit][cont]['baseline']), np.mean(data[unit][cont]['baseline'])],'ko--') 
                    ax2.plot([surr_narrow_new,surr_narrow_new],[np.mean(data[unit][cont]['baseline']), np.max(mean_container)],'g--')
                    # plot fitted curve
                    ax2.plot(diams_tight,Rhat,'k-')
                    ax2.set_xscale('log')
                    ax2.set_xlabel('Diameter (deg)')
                    ax2.set_ylabel('Spike-count')

                    ax2b.set_xscale('log')
                    ax2b.errorbar(data[unit]['info']['diam'], fano_container, yerr=fano_ERR_container, fmt=f_fano, label='fano')
                    ax2b.plot([data[unit]['info']['diam'][0], data[unit]['info']['diam'][-1]],[np.mean(bsl_container), np.mean(bsl_container)],f_bsl)
                    ax2b.plot(diams_tight, Fhat, f_fano_model)
                    ax2b.set_ylabel('Fano-factor')
                    plt.title('100ms count window')
                    plt.legend()

                    # collect parameters from the fits and set bg to gray for shitty fits
                    if unit in excluded_fits:
                        ax2.set_facecolor((0.5, 0.5, 0.5))
                        fit_fano_SML = np.nan
                        fit_fano_RF  = np.nan
                        fit_fano_SUR = np.nan
                        fit_fano_LAR = np.nan
                        fit_fano_BSL = np.nan
                        fit_fano_MIN = np.nan
                        fit_fano_MAX = np.nan

                        fit_fano_MAX_diam = np.nan
                        fit_fano_MIN_diam = np.nan
                        fit_RF            = np.nan
                        fit_surr          = np.nan
                    else:
                        ax2.set_facecolor((1.0, 1.0, 1.0))
                        fit_fano_SML = Fhat[0]
                        fit_fano_RF  = Fhat[np.argmax(Rhat)]
                        fit_fano_SUR = Fhat[surr_ind_narrow_new]
                        fit_fano_LAR = Fhat[-1]
                        fit_fano_BSL = np.mean(bsl_container)
                        fit_fano_MIN = np.min(Fhat)
                        fit_fano_MAX = np.max(Fhat)

                        fit_fano_MAX_diam = diams_tight[np.argmax(Fhat)]
                        fit_fano_MIN_diam = diams_tight[np.argmin(Fhat)]
                        fit_RF            = diams_tight[np.argmax(Rhat)]
                        fit_surr          = diams_tight[surr_ind_narrow_new]



                    ##########################
                    # 2nd row
                    mx_ind = np.argmax(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1))
                    
                    # center fano-vs-firing
                    ax5 = plt.subplot(4,4,5)
                    ax5.set_xscale('log')
                    ax5.set_yscale('log')
                    ax5.plot(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)[0:mx_ind+1], np.squeeze(fano)[0:mx_ind+1], 'ko')
                    ax5.set_ylabel('Fano-factor')
                    ax5.set_xlabel('Spike-count')

                    # linear fit 
                    spk_mn = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)[0:mx_ind+1]
                    spk_fn = np.squeeze(fano)[0:mx_ind+1]
                    dd2 = np.zeros((spk_mn.shape[0],2))
                    dd2[:,0] = spk_fn
                    dd2[:,1] = spk_mn

                    df = pd.DataFrame(data=dd2, columns=['FANO','MEAN'])
                    lm = ols('FANO ~ MEAN',data=df).fit()
                    center_slope_fano = lm.params['MEAN']

                    ax5.plot([np.min(spk_mn), np.max(spk_mn)], lm.params['Intercept'] + lm.params['MEAN']*np.array([np.min(spk_mn), np.max(spk_mn)]), 'k-')
                    
                    # center fano-vs-firing short count window
                    ax6 = plt.subplot(4,4,6)
                    ax6.set_xscale('log')
                    ax6.set_yscale('log')
                    ax6.plot(mean_container[0:max_ind+1], fano_container[0:max_ind+1], 'ko')
                    ax6.set_ylabel('Fano-factor')
                    ax6.set_xlabel('Spike-count')
                    # linear fit 
                    spk_mn = mean_container[0:max_ind+1]
                    spk_vr = fano_container[0:max_ind+1]
                    dd2 = np.zeros((spk_mn.shape[0],2))
                    dd2[:,0] = spk_vr
                    dd2[:,1] = spk_mn

                    df = pd.DataFrame(data=dd2, columns=['VAR','MEAN'])
                    lm = ols('VAR ~ MEAN',data=df).fit()
                    center_slope = lm.params['MEAN']
                    
                    ax6.plot([np.min(spk_mn), np.max(spk_mn)], lm.params['Intercept'] + lm.params['MEAN']*np.array([np.min(spk_mn), np.max(spk_mn)]), 'k-')

                    ##########################
                    # 3rd row
                    mx_ind = np.argmax(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1))

                    # surround fano-vs-firing
                    ax9 = plt.subplot(4,4,9)
                    ax9.set_xscale('log')
                    ax9.set_yscale('log')
                    ax9.plot(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)[mx_ind:surr_ind+1], np.squeeze(fano)[mx_ind:surr_ind+1], 'ko')
                    ax9.set_ylabel('Fano-factor')
                    ax9.set_xlabel('Spike-count')
                    # linear fit 
                    spk_mn = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)[mx_ind:surr_ind+1]
                    spk_fn = np.squeeze(fano)[mx_ind:surr_ind+1]
                    dd2 = np.zeros((spk_mn.shape[0],2))
                    dd2[:,0] = spk_fn
                    dd2[:,1] = spk_mn

                    df = pd.DataFrame(data=dd2, columns=['FANO','MEAN'])
                    lm = ols('FANO ~ MEAN',data=df).fit()
                    surround_slope_fano = lm.params['MEAN']

                    ax9.plot([np.min(spk_mn), np.max(spk_mn)], lm.params['Intercept'] + lm.params['MEAN']*np.array([np.min(spk_mn), np.max(spk_mn)]), 'k-')
                    
                    # surround fano-vs-firing short count window
                    ax10 = plt.subplot(4,4,10)
                    ax10.set_xscale('log')
                    ax10.set_yscale('log')
                    ax10.plot(mean_container[max_ind:surr_ind_narrow+1], fano_container[max_ind:surr_ind_narrow+1], 'ko')
                    ax10.set_ylabel('Fano-factor')
                    ax10.set_xlabel('Spike-count')
                    # linear fit
                    spk_mn = mean_container[max_ind:surr_ind_narrow+1]
                    spk_vr = fano_container[max_ind:surr_ind_narrow+1]
                    dd2 = np.zeros((spk_mn.shape[0],2))
                    dd2[:,0] = spk_vr
                    dd2[:,1] = spk_mn

                    df = pd.DataFrame(data=dd2, columns=['VAR','MEAN'])
                    lm = ols('VAR ~ MEAN',data=df).fit()
                    surround_slope = lm.params['MEAN']

                    ax10.plot([np.min(spk_mn), np.max(spk_mn)], lm.params['Intercept'] + lm.params['MEAN']*np.array([np.min(spk_mn), np.max(spk_mn)]), 'k-')
                    
                    plt.tight_layout()
                    examples_pdf.savefig()

                    ax1.cla()
                    ax1b.cla()
                    ax2.cla()
                    ax2b.cla()
                    ax5.cla()
                    ax6.cla()
                    ax9.cla()
                    ax10.cla()
                    
                    spkC = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)
                    if center_slope <=0:
                        centerSIG = -1
                    else:
                        centerSIG = 1

                    if surround_slope_fano <=0:
                        surroundSIG_fano = -1
                    else:
                        surroundSIG_fano = 1

                    

                    # place unit parameters to a dataframe for later analysis
                    para_tmp = np.ones((1,10),dtype=object)*np.nan
                    para_tmp = {'RFdiam':data[unit]['info']['diam'][mx_ind],
                                'maxResponse':np.max(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)),
                                'SI':(np.max(spkC) - spkC[-1]) / np.max(spkC),
                                'baseline':np.mean(data[unit][100.0]['baseline']),
                                'layer':L,
                                'anipe':anipe,
                                'center_slope':center_slope,
                                'surround_slope':surround_slope,
                                'centerSIG':centerSIG,
                                'center_slope_fano':center_slope_fano,
                                'surround_slope_fano':surround_slope_fano,
                                'ntrials':ntrials,
                                'surroundSIG_fano':surroundSIG_fano,
                                'center_slope_narrow_window':center_slope_narrow_window,
                                'surround_slope_narrow_window':surround_slope_narrow_window,
                                'fit_fano_SML':fit_fano_SML,
                                'fit_fano_RF':fit_fano_RF,
                                'fit_fano_SUR':fit_fano_SUR,
                                'fit_fano_LAR':fit_fano_LAR,
                                'fit_fano_BSL':fit_fano_BSL,
                                'fit_fano_MIN':fit_fano_MIN,
                                'fit_fano_MAX':fit_fano_MAX,
                                'fit_fano_MIN_diam':fit_fano_MIN_diam,
                                'fit_fano_MAX_diam':fit_fano_MAX_diam}

                    tmp_df = pd.DataFrame(para_tmp, index=[indx])
                    params_df = params_df.append(tmp_df,sort=True)
                    indx = indx + 1
                    
                    if plot_rasters:
                        plt.figure(2)
                        # rasters
                        for a in range(data[unit]['info']['diam'].shape[0]):
                            spks = data[unit][cont]['spkR_NoL'][:,a,:]
                            ax = plt.subplot(7,3,a+1)
                            dalib.rasters(spks, bins, ax)

                        plt.tight_layout()
                        examples_pdf.savefig()
                        plt.clf()

examples_pdf.close()
params_df.to_csv(S_dir+'extracted_params_Dec-2021.csv')

# save data
with open(S_dir + 'mean_PSTHs-MK-MU.pkl','wb') as f:
    pkl.dump(mean_PSTHs,f,pkl.HIGHEST_PROTOCOL)

with open(S_dir + 'vari_PSTHs-MK-MU.pkl','wb') as f:
    pkl.dump(vari_PSTHs,f,pkl.HIGHEST_PROTOCOL)

# layer resolved
# SG
with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','wb') as f:
    pkl.dump(mean_PSTHs_SG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'vari_PSTHs_SG-MK-MU.pkl','wb') as f:
    pkl.dump(vari_PSTHs_SG,f,pkl.HIGHEST_PROTOCOL)
# G
with open(S_dir + 'mean_PSTHs_G-MK-MU.pkl','wb') as f:
    pkl.dump(mean_PSTHs_G,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'vari_PSTHs_G-MK-MU.pkl','wb') as f:
    pkl.dump(vari_PSTHs_G,f,pkl.HIGHEST_PROTOCOL)
# IG
with open(S_dir + 'mean_PSTHs_IG-MK-MU.pkl','wb') as f:
    pkl.dump(mean_PSTHs_IG,f,pkl.HIGHEST_PROTOCOL)
with open(S_dir + 'vari_PSTHs_IG-MK-MU.pkl','wb') as f:
    pkl.dump(vari_PSTHs_IG,f,pkl.HIGHEST_PROTOCOL)
 
