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

examples_SG = PdfPages(S_dir + 'correlations-candidates-SUPRAGRANULAR.pdf')
examples_G  = PdfPages(S_dir + 'correlations-candidates-GRANULAR.pdf')
examples_IG = PdfPages(S_dir + 'correlations-candidates-INFRAGRANULAR.pdf')
#examples_MX = PdfPages(S_dir + 'correlations-candidates-MX.pdf')

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
            
            u1_baseline = np.mean(np.sum(data[pair][cont]['spkR_NoL_pair1'][:,:,bsl_begin:bsl_end],axis=2))
            u2_baseline = np.mean(np.sum(data[pair][cont]['spkR_NoL_pair2'][:,:,bsl_begin:bsl_end],axis=2))
            u1_responsive = select_data(data[pair][cont]['spkC_NoL_pair1'].T,u1_baseline)
            u2_responsive = select_data(data[pair][cont]['spkC_NoL_pair2'].T,u2_baseline)

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

                plt.figure(1,figsize=(14,12))

                # plot rasters, correlation data and fit, spike-count scatter plots, 
                f, ax = plt.subplots(1,3,num=1)

                # plot data for baseline
                # --------------------------------------------------
                plot_colm = 0                
                plot_diam = 0
                                
                # get z-scored spike-counts
                sc1 = np.sum(np.squeeze(data[pair][cont]['spkR_NoL_pair1'][:,plot_diam,bsl_begin:bsl_end]),axis=1)
                sc2 = np.sum(np.squeeze(data[pair][cont]['spkR_NoL_pair2'][:,plot_diam,bsl_begin:bsl_end]),axis=1)
                z_sc1 = dalib.z_score(sc1)
                z_sc2 = dalib.z_score(sc2)
                model = sm.OLS(z_sc2,z_sc1).fit()
                # plot z-scored spike-counts
                ax[plot_colm].scatter(z_sc1,z_sc2,color='k',s=5)
                ax[plot_colm].plot(np.linspace(-3,3,100),np.linspace(-3,3,100)*model.params[0],'k-')
                ax[plot_colm].set_xlim([-3,3])
                ax[plot_colm].set_ylim([-3,3])
                ax[plot_colm].set_aspect('equal',adjustable='box')
                ax[plot_colm].set_title(str(pair)+' '+str(model.params[0]))
                # --------------------------------------------------

                # plot data for RF
                # --------------------------------------------------
                plot_diam = np.argmax(gm_response)                
                plot_colm = 1

                # get z-scored spike-counts
                z_sc1 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair1'][:,plot_diam])
                z_sc2 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair2'][:,plot_diam])
                model = sm.OLS(z_sc2,z_sc1).fit()
                # plot z-scored spike-counts
                ax[plot_colm].scatter(z_sc1,z_sc2,color='k',s=5)
                ax[plot_colm].plot(np.linspace(-3,3,100),np.linspace(-3,3,100)*model.params[0],'k-')
                ax[plot_colm].set_xlim([-3,3])
                ax[plot_colm].set_ylim([-3,3])
                ax[plot_colm].set_aspect('equal',adjustable='box')
                ax[plot_colm].set_title(str(model.params[0]))
                # --------------------------------------------------

                # plot data for Largest diameter
                # --------------------------------------------------
                plot_diam = -2
                plot_colm = 2
                                
                # get z-scored spike-counts
                z_sc1 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair1'][:,plot_diam])
                z_sc2 = dalib.z_score(data[pair][100.0]['spkC_NoL_pair2'][:,plot_diam])
                model = sm.OLS(z_sc2,z_sc1).fit()
                # plot z-scored spike-counts
                ax[plot_colm].scatter(z_sc1,z_sc2,color='k',s=5)
                ax[plot_colm].plot(np.linspace(-3,3,100),np.linspace(-3,3,100)*model.params[0],'k-')
                ax[plot_colm].set_xlim([-3,3])
                ax[plot_colm].set_ylim([-3,3])
                ax[plot_colm].set_aspect('equal',adjustable='box')
                ax[plot_colm].set_title(str(model.params[0]))
                # --------------------------------------------------

                # determine layer type and save to PDF accordinly
                if data[pair]['info']['L1'] == 'LSG' and data[pair]['info']['L2'] == 'LSG':
                    examples_SG.savefig()
                elif data[pair]['info']['L1'] == 'L4C' and data[pair]['info']['L2'] == 'L4C':
                    examples_G.savefig()
                elif data[pair]['info']['L1'] == 'LIG' and data[pair]['info']['L2'] == 'LIG':
                    examples_IG.savefig()
                else:
                    print('Error: Layer types do not match')
                    #examples_MX.savefig()
                
                plt.clf()

#
examples_SG.close()
examples_G.close()
examples_IG.close()
#examples_MX.close()