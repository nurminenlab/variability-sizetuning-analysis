import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.optimize import basinhopping, curve_fit
import scipy.io as scio

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
mat_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/PSTHmats/'

MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_mn_data = pkl.load(f)

with open(S_dir + 'vari_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_vr_data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_G-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)


diams = np.array(list(diams_data.keys()))
del(diams_data)

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

def cost_fano(params,xdata,ydata):
    Rhat = dalib.doubleROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

eps = 0.0000001
# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

fano_PSTH_first_tp = 300

# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100
nboots = 3000

unit = 51
mn_mtrx = G_mn_data[unit]
vr_mtrx = G_vr_data[unit]

fano      = np.nan * np.ones((mn_mtrx.shape[0]))
FR        = np.nan * np.ones((mn_mtrx.shape[0]))
fano_bsl  = np.nan * np.ones((mn_mtrx.shape[0]))
FR_bsl    = np.nan * np.ones((mn_mtrx.shape[0]))
FR_boot   = np.nan * np.ones((nboots,mn_mtrx.shape[0]))
fano_boot = np.nan * np.ones((nboots,mn_mtrx.shape[0]))
fano_PSTH_RF = np.nan * np.ones((nboots,data[unit][cont]['spkR_NoL'][:,0,:].shape[1] - fano_PSTH_first_tp))

if mn_mtrx.shape[0] == 18:
    diamsa = diams[1:]
else:
    diamsa = diams

a = 0

for stim in range(mn_mtrx.shape[0]):

    fano[stim] = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
    FR[stim]   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)/(count_window/1000)
        
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
    fano_boot[:,stim] = np.mean(np.divide(vari_PSTH_booted[:,first_tp:last_tp],
                                        (eps + mean_PSTH_booted[:,first_tp:last_tp])),axis=1)

    
    # Firing-rate time-course
    FR_boot[:,stim] = np.mean(mean_PSTH_booted[:,first_tp:last_tp],axis=1)/(count_window/1000)

    if stim == 4 or stim == 18:

        fano_PSTH_RF = np.divide(vari_PSTH_booted[:,fano_PSTH_first_tp:], (eps + mean_PSTH_booted[:,fano_PSTH_first_tp:]))
        fano_PSTH_RF_SD = np.std(fano_PSTH_RF,axis=0)

        # convert to Hz
        PSTH_RF = mean_PSTH_booted[:,fano_PSTH_first_tp:]/(count_window/1000)
        PSTH_RF_SD = np.std(PSTH_RF,axis=0)
        
        fano_bsl[stim] = np.mean(vr_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window] / (eps + mn_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window]))
        FR_bsl[stim]   = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)/(count_window/1000)

        plt.figure(1,figsize=(1.335, 1.115))
        ax = plt.subplot(2,1,a+1)
        axb = ax.twinx()
        t = np.arange(-100,600,1)
        # plot fano-PSTH
        ax.fill_between(t,np.mean(fano_PSTH_RF,axis=0) - fano_PSTH_RF_SD, np.mean(fano_PSTH_RF,axis=0) + fano_PSTH_RF_SD,color='red')
        ax.plot(t,np.mean(fano_PSTH_RF,axis=0), '-',color=[0.5, 0, 0])
        ax.plot([-100,600],[np.mean(fano_PSTH_RF[:,0:100]),np.mean(fano_PSTH_RF[:,0:100])], '--',color='red')
        ax.set_ylim(0,5)
        ax.tick_params(axis='y',color='red')
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y',colors='red',labelsize=8)
        ax.tick_params(axis='x',labelsize=8)
        # ax.set_ylabel('Fano-factor')
        # if a == 1:
        #     ax.set_xlabel('Peri-stimulus time (ms)')

        ax.yaxis.label.set_color('red')
        ax.spines['top'].set_visible(False)

        # plot PSTH
        axb.fill_between(t,np.mean(PSTH_RF,axis=0) - PSTH_RF_SD, np.mean(PSTH_RF,axis=0) + PSTH_RF_SD,color='gray')
        axb.plot(t,np.mean(PSTH_RF,axis=0), 'k-')
        axb.plot([-100,600],[np.mean(PSTH_RF[:,0:100]),np.mean(PSTH_RF[:,0:100])], 'k--')
        axb.set_ylim(0,150)        
        axb.spines['left'].set_color('red')
        axb.spines['top'].set_visible(False)
        axb.tick_params(axis='y',labelsize=8)
        #axb.set_ylabel('Firing-rate (Hz)')

        plt.figure(2)
        ax = plt.subplot(2,1,1)
        t = np.arange(50,450,1)
        # plot fano-PSTH
        ax.fill_between(t,np.mean(fano_PSTH_RF[:,150:550],axis=0) - fano_PSTH_RF_SD[150:550], np.mean(fano_PSTH_RF[:,150:550],axis=0) + fano_PSTH_RF_SD[150:550],color='red')
        ax.plot(t,np.mean(fano_PSTH_RF[:,150:550],axis=0), '-',color=[0.5, 0, 0])
        ax.set_ylabel('Fano-factor')
        ax.set_xlabel('Peri-stimulus time (ms)')
        ax.spines['left'].set_color('red')
        ax.tick_params(axis='y',colors='red')
        ax.yaxis.label.set_color('red')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.figure(3,figsize=(1.335, 1.115))
        # plot rasters
        t = np.arange(-100,600,1)
        spkR = data[unit][cont]['spkR_NoL'][:,stim,fano_PSTH_first_tp:] > 0
        ax = plt.subplot(2,1,a+1)
        for tr in range(spkR.shape[0]):
            ax.vlines(t[spkR[tr,:]],ymin=0+tr,ymax=1+tr,linewidth=0.1,color='k')
            #ax.plot(t[spkR[tr,:]],np.ones(t[spkR[tr,:]].shape[0])+tr,'k.',markersize=0.1)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([0, 250, 500])
        ax.tick_params(axis='x',labelsize=8)
        ax.tick_params(axis='y',labelsize=8)
        # ax.set_ylabel('Trial')
        # ax.set_xlabel('Peri-stimulus time (ms)')
                
        a +=1
        
        

plt.figure(1)
plt.savefig(fig_dir + 'F1_G_PSTH_fanoPSTH.svg',bbox_inches='tight',pad_inches=0)        
plt.figure(2)
plt.savefig(fig_dir + 'F1_G_fano-PSTH-zoomed.svg',bbox_inches='tight',pad_inches=0)
plt.figure(3)
plt.savefig(fig_dir + 'F1_G_rasters.eps',bbox_inches='tight',pad_inches=0)

##
fano_E = 2 * np.std(fano_boot,axis=0)
FR_E   = 2 * np.std(FR_boot,axis=0)

# ROG fit spike-count data 
try:
    popt,pcov = curve_fit(dalib.ROG,diamsa,FR,bounds=(0,np.inf),maxfev=100000)
except:
    args = (diamsa,FR)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x

args = (diamsa,fano)
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args},seed=1234,niter=1000)
diams_tight = np.logspace(np.log10(diamsa[0]),np.log10(diamsa[-1]),1000)
Rhat = dalib.ROG(diams_tight,*popt)
Fhat = dalib.doubleROG(diams_tight,*res.x)

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

RFsurr = diams_tight[surr_ind_narrow_new]
RFsize = diams_tight[np.argmax(Rhat)]
SI = (np.max(Rhat) - Rhat[-1]) / np.max(Rhat)

plt.figure(4,figsize=(1.335, 1.115))
ax = plt.subplot(1,1,1)
axb = ax.twinx()
ax.errorbar(diamsa, fano, yerr=fano_E,fmt='ro',markersize=4,mfc='None',lw=1)
axb.errorbar(diamsa, FR, yerr=FR_E,fmt='ko',markersize=4,mfc='None',lw=1)
ax.plot([diams[0],diams[-1]],[np.nanmean(fano_bsl),np.nanmean(fano_bsl)],'r--')
axb.plot([diams[0],diams[-1]],[np.nanmean(FR_bsl),np.nanmean(FR_bsl)],'k--')
# fits
ax.plot(diams_tight, Fhat, 'r-',lw=1)
axb.plot(diams_tight, Rhat, 'k-',lw=1)
ax.set_xscale('log')
axb.set_xscale('log')

ax.spines['left'].set_color('red')
axb.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red',labelsize=8)
axb.tick_params(axis='y',labelsize=8)
axb.tick_params(axis='x',labelsize=8)
ax.tick_params(axis='x',labelsize=8)
# ax.set_ylabel('Fano-factor')
# ax.set_xlabel('Stimulus diameter')
ax.yaxis.label.set_color('red')
ax.spines['top'].set_visible(False)
axb.spines['top'].set_visible(False)

plt.figure(4)
plt.savefig(fig_dir + 'F1_G_ASFs.svg',bbox_inches='tight',pad_inches=0)
