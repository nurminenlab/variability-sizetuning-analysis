import sys
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
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

examples_SG = PdfPages(S_dir + 'correlations-individualunits-SUPRAGRANULAR.pdf')
examples_G  = PdfPages(S_dir + 'correlations-individualunits-GRANULAR.pdf')
examples_IG = PdfPages(S_dir + 'correlations-individualunits-INFRAGRANULAR.pdf')
examples_MX = PdfPages(S_dir + 'correlations-individualunits-MX.pdf')


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
bins = np.arange(-400,600,1)
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
                a0 = data[pair][cont]['corr_bin_NoL'].shape[0]
                a1 = data[pair][cont]['corr_bin_NoL'].shape[1]
                a2 = data[pair][cont]['corr_bin_NoL_booted'].shape[1]
                
                corrs_mean = np.reshape(data[pair][cont]['corr_bin_NoL'],(a0,a1))
                corrs_err = np.mean(np.percentile(np.reshape(data[pair][cont]['corr_bin_NoL_booted'],(a0,a2)),[16,85],axis=1).T,axis=1)
                args = (data[pair]['info']['diam'],corrs_mean[:,0])
                
                bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,-1,-1],[1,30,30,30,100,100,100,100,1,1]]).T
                res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
                Corr_hat = dalib.doubleROG(diams_tight,*res.x)

                # plot rasters, correlation data and fit, spike-count scatter plots, 
                f, ax = plt.subplots(4,4,figsize=(14,12))

                # plot data for peak correlation
                # --------------------------------------------------
                plot_diam = np.argmin(np.abs(data[pair]['info']['diam'] - diams_tight[np.argmax(Corr_hat)]))
                plot_diam_ind_Cmax = plot_diam
                plot_colm = 0
                dalib.rasters(np.squeeze(data[pair][cont]['spkR_NoL_pair1'][:,plot_diam,:]), bins, ax[0,plot_colm],color='red')
                dalib.rasters(np.squeeze(data[pair][cont]['spkR_NoL_pair2'][:,plot_diam,:]), bins, ax[1,plot_colm],color='blue')
                
                # get PSTHs for both units
                mean_PSTH_u1, vari_PSTH,binned_data,mean_PSTH_booted_u1, vari_PSTH_booted = dalib.meanvar_PSTH(data[pair][cont]['spkR_NoL_pair1'][:,plot_diam,bsl_begin:],
                                                                                                                 count_window,
                                                                                                                 style='same',
                                                                                                                 return_bootdstrs=True,
                                                                                                                 nboots=1000)

                mean_PSTH_u2, vari_PSTH,binned_data,mean_PSTH_booted_u2, vari_PSTH_booted = dalib.meanvar_PSTH(data[pair][cont]['spkR_NoL_pair2'][:,plot_diam,bsl_begin:],
                                                                                                                 count_window,
                                                                                                                 style='same',
                                                                                                                 return_bootdstrs=True,
                                                                                                                 nboots=1000)
                # plot PSTHs
                ax[2,plot_colm].fill_between(bins[bsl_begin:],mean_PSTH_u1 - np.std(mean_PSTH_booted_u1,axis=0), 
                                    mean_PSTH_u1 + np.std(mean_PSTH_booted_u1,axis=0),color='red',alpha=0.5)
                ax[2,plot_colm].plot(bins[bsl_begin:],mean_PSTH_u1,'k-')
                ax[2,plot_colm].fill_between(bins[bsl_begin:],mean_PSTH_u2 - np.std(mean_PSTH_booted_u2,axis=0), 
                                    mean_PSTH_u2 + np.std(mean_PSTH_booted_u2,axis=0),color='blue',alpha=0.5)
                ax[2,plot_colm].plot(bins[bsl_begin:],mean_PSTH_u2,'b-')

                # get z-scored spike-counts
                z_sc1 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair1'][:,plot_diam])
                z_sc2 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair2'][:,plot_diam])
                # plot z-scored spike-counts
                ax[3,plot_colm].scatter(z_sc1,z_sc2,color='k',s=5)
                ax[3,plot_colm].set_xlim([-3,3])
                ax[3,plot_colm].set_ylim([-3,3])
                ax[3,plot_colm].set_aspect('equal',adjustable='box')
                # --------------------------------------------------

                # plot data for RF
                # --------------------------------------------------
                plot_diam = np.argmax(gm_response)                
                plot_colm = 1
                dalib.rasters(np.squeeze(data[pair][cont]['spkR_NoL_pair1'][:,plot_diam,:]), bins, ax[0,plot_colm],color='red')
                dalib.rasters(np.squeeze(data[pair][cont]['spkR_NoL_pair2'][:,plot_diam,:]), bins, ax[1,plot_colm],color='blue')
                
                # get PSTHs for both units
                mean_PSTH_u1, vari_PSTH,binned_data,mean_PSTH_booted_u1, vari_PSTH_booted = dalib.meanvar_PSTH(data[pair][cont]['spkR_NoL_pair1'][:,plot_diam,bsl_begin:],
                                                                                                                 count_window,
                                                                                                                 style='same',
                                                                                                                 return_bootdstrs=True,
                                                                                                                 nboots=1000)

                mean_PSTH_u2, vari_PSTH,binned_data,mean_PSTH_booted_u2, vari_PSTH_booted = dalib.meanvar_PSTH(data[pair][cont]['spkR_NoL_pair2'][:,plot_diam,bsl_begin:],
                                                                                                                 count_window,
                                                                                                                 style='same',
                                                                                                                 return_bootdstrs=True,
                                                                                                                 nboots=1000)
                # plot PSTHs
                ax[2,plot_colm].fill_between(bins[bsl_begin:],mean_PSTH_u1 - np.std(mean_PSTH_booted_u1,axis=0), 
                                    mean_PSTH_u1 + np.std(mean_PSTH_booted_u1,axis=0),color='red',alpha=0.5)
                ax[2,plot_colm].plot(bins[bsl_begin:],mean_PSTH_u1,'k-')
                ax[2,plot_colm].fill_between(bins[bsl_begin:],mean_PSTH_u2 - np.std(mean_PSTH_booted_u2,axis=0), 
                                    mean_PSTH_u2 + np.std(mean_PSTH_booted_u2,axis=0),color='blue',alpha=0.5)
                ax[2,plot_colm].plot(bins[bsl_begin:],mean_PSTH_u2,'b-')

                # get z-scored spike-counts
                z_sc1 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair1'][:,plot_diam])
                z_sc2 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair2'][:,plot_diam])
                # plot z-scored spike-counts
                ax[3,plot_colm].scatter(z_sc1,z_sc2,color='k',s=5)
                ax[3,plot_colm].set_xlim([-3,3])
                ax[3,plot_colm].set_ylim([-3,3])
                ax[3,plot_colm].set_aspect('equal',adjustable='box')
                # --------------------------------------------------

                # plot data for Largest diameter
                # --------------------------------------------------
                plot_diam = -1
                plot_colm = 2
                dalib.rasters(np.squeeze(data[pair][cont]['spkR_NoL_pair1'][:,plot_diam,:]), bins, ax[0,plot_colm],color='red')
                dalib.rasters(np.squeeze(data[pair][cont]['spkR_NoL_pair2'][:,plot_diam,:]), bins, ax[1,plot_colm],color='blue')
                
                # get PSTHs for both units
                mean_PSTH_u1, vari_PSTH,binned_data,mean_PSTH_booted_u1, vari_PSTH_booted = dalib.meanvar_PSTH(data[pair][cont]['spkR_NoL_pair1'][:,plot_diam,bsl_begin:],
                                                                                                                 count_window,
                                                                                                                 style='same',
                                                                                                                 return_bootdstrs=True,
                                                                                                                 nboots=1000)

                mean_PSTH_u2, vari_PSTH,binned_data,mean_PSTH_booted_u2, vari_PSTH_booted = dalib.meanvar_PSTH(data[pair][cont]['spkR_NoL_pair2'][:,plot_diam,bsl_begin:],
                                                                                                                 count_window,
                                                                                                                 style='same',
                                                                                                                 return_bootdstrs=True,
                                                                                                                 nboots=1000)
                # plot PSTHs
                ax[2,plot_colm].fill_between(bins[bsl_begin:],mean_PSTH_u1 - np.std(mean_PSTH_booted_u1,axis=0), 
                                    mean_PSTH_u1 + np.std(mean_PSTH_booted_u1,axis=0),color='red',alpha=0.5)
                ax[2,plot_colm].plot(bins[bsl_begin:],mean_PSTH_u1,'k-')
                ax[2,plot_colm].fill_between(bins[bsl_begin:],mean_PSTH_u2 - np.std(mean_PSTH_booted_u2,axis=0), 
                                    mean_PSTH_u2 + np.std(mean_PSTH_booted_u2,axis=0),color='blue',alpha=0.5)
                ax[2,plot_colm].plot(bins[bsl_begin:],mean_PSTH_u2,'b-')

                # get z-scored spike-counts
                z_sc1 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair1'][:,plot_diam])
                z_sc2 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair2'][:,plot_diam])
                # plot z-scored spike-counts
                ax[3,plot_colm].scatter(z_sc1,z_sc2,color='k',s=5)
                ax[3,plot_colm].set_xlim([-3,3])
                ax[3,plot_colm].set_ylim([-3,3])
                ax[3,plot_colm].set_aspect('equal',adjustable='box')
                # --------------------------------------------------

                
                # plot geometric mean response and correlations + fits
                # --------------------------------------------------
                plot_colm = 3                 
                ax1b = ax[0,plot_colm].twinx()
                ax[0,plot_colm].errorbar(data[pair]['info']['diam'],gm_response,yerr=gm_response_SE,fmt='ko',label='spike-count',ms=5)
                ax[0,plot_colm].plot([data[pair]['info']['diam'][0],data[pair]['info']['diam'][-1]],[gm_baseline, gm_baseline],'ko--') 
                # fitted curve
                ax[0,plot_colm].plot(diams_tight,gm_Rhat,'k-')

                ax1b.set_xscale('log')
                ax1b.errorbar(data[pair]['info']['diam'], corrs_mean, yerr=corrs_err, fmt=c_corr, label='corr',ms=5)
                ax1b.plot([data[pair]['info']['diam'][0], data[pair]['info']['diam'][-1]],[np.mean(bsl_container), np.mean(bsl_container)],c_bsl)
                ax1b.plot(diams_tight, Corr_hat, c_corr_model)
                ax1b.plot(data[pair]['info']['diam'][plot_diam_ind_Cmax],corrs_mean[plot_diam_ind_Cmax],'k*',ms=5,zorder=10)
                ax1b.tick_params(axis='y',color='green')
                ax1b.spines['left'].set_color('green')
                ax1b.tick_params(axis='y',colors='green')
                ax1b.set_ylabel('rSC')

                ax[0,plot_colm].set_xscale('log')
                ax[0,plot_colm].set_xlabel('Diameter (deg)')
                ax[0,plot_colm].set_ylabel('Spike-count')
                # --------------------------------------------------

                # extract correlations from the fits at certain diameters, as done in the paper
                # --------------------------------------------------
                plot_colm = 3
                C = np.array([np.mean(bsl_container),np.max(Corr_hat),Corr_hat[np.argmax(gm_Rhat)],Corr_hat[-1]])
                ax[1,plot_colm].bar([1,2,3,4],C,ec='black',fc='gray',width=1)
                
                # determine layer type and save to PDF accordinly
                if data[pair]['info']['L1'] == 'LSG' and data[pair]['info']['L2'] == 'LSG':
                    examples_SG.savefig()
                elif data[pair]['info']['L1'] == 'L4C' and data[pair]['info']['L2'] == 'L4C':
                    examples_G.savefig()
                elif data[pair]['info']['L1'] == 'LIG' and data[pair]['info']['L2'] == 'LIG':
                    examples_IG.savefig()
                else:
                    examples_MX.savefig()

#
examples_SG.close()
examples_G.close()
examples_IG.close()
examples_MX.close()
