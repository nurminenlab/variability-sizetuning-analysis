import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import scipy.stats as sts
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')

#import pdb
import data_analysislib as dalib
import pandas as pd
import scipy as sc
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import basinhopping, curve_fit
from statsmodels.formula.api import ols

plot_rasters = False
cont_wndw_length = 100
boot_num = int(1e3)

plotter = 'contrast'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'

SUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

examples_AL = PdfPages(S_dir + 'fanofactors-individualunits-ALL.pdf')
examples_SG = PdfPages(S_dir + 'fanofactors-individualunits-SUPRAGRANULAR.pdf')
examples_G  = PdfPages(S_dir + 'fanofactors-individualunits-GRANULAR.pdf')
examples_IG = PdfPages(S_dir + 'fanofactors-individualunits-INFRAGRANULAR.pdf')

# for these units we will use constrained optimization
shitty_fits = [0,1,2,3,5,6,7,13,19,35,58,53,57,59,68,70,71,72,77,78,79,90,92,98,105]
# we were not able to fit this unit and excluded it from the analysis
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
eps = 0.0000001
raster_start = 300

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
                    
                    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
                    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
                    
                    Fhat = dalib.doubleROG(diams_tight,*res.x)                   

                    ##########################
                    surr,surr_ind = dalib.saturation_point(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1),data[unit]['info']['diam'],criteria=1.3)
                    ntrials = data[unit][cont]['spkC_NoL'].shape[0]
                    mx_ind = np.argmax(np.mean(data[unit][cont]['spkC_NoL'].T, axis=1))
                    spkC = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)                    
                    
                    response    = np.mean(data[unit][cont]['spkC_NoL'].T, axis=1)
                    response_SE = np.std(data[unit][cont]['spkC_NoL'].T, axis=1) / np.sqrt(data[unit][cont]['spkC_NoL'].T.shape[1])

                    # plot rasters, correlation data and fit, spike-count scatter plots, 
                    f, ax = plt.subplots(1,1,num=1)
                    
                    axb = ax.twinx()
                    axb.plot(data[unit]['info']['diam'],mean_container,'ko')
                    axb.plot(diams_tight, Rhat, 'k-')
                    ax.plot(data[unit]['info']['diam'],fano_container,'ro')
                    ax.plot(diams_tight, Fhat, 'r-')
                    ax.set_title(str(unit))
                    ax.set_xscale('log') 


                    examples_AL.savefig()     
                    # determine layer type and save to PDF accordinly
                    if data[unit]['info']['layer'].decode('utf-8') == 'LSG':
                        examples_SG.savefig()
                    elif data[unit]['info']['layer'].decode('utf-8') == 'L4C':
                        examples_G.savefig()
                    elif data[unit]['info']['layer'].decode('utf-8') == 'LIG':
                        examples_IG.savefig()
                    else:
                        print('This should not happen!')
                
                    plt.clf()
                    # blah
examples_AL.close()
examples_SG.close()
examples_G.close()
examples_IG.close()