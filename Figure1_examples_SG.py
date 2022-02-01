import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/DataAnalysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'

MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_mn_data = pkl.load(f)

with open(S_dir + 'vari_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_vr_data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)


diams = np.array(list(diams_data.keys()))
del(diams_data)


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

unit = 71
mn_mtrx = SG_mn_data[unit]
vr_mtrx = SG_vr_data[unit]

fano      = np.nan * np.ones((mn_mtrx.shape[0]))
FR        = np.nan * np.ones((mn_mtrx.shape[0]))
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

    # bootstrapped fano time-course
    fano_boot[:,stim] = np.mean(np.divide(vari_PSTH_booted[:,fano_PSTH_first_tp:last_tp], 
                                (eps + mean_PSTH_booted[:,fano_PSTH_first_tp:last_tp])),axis=1)
    FR_boot[:,stim] = np.mean(mean_PSTH_booted[:,first_tp:last_tp],axis=1)/(count_window/1000)

    if stim == 2 or stim == 18:
        
        fano_PSTH_RF = np.divide(vari_PSTH_booted[:,fano_PSTH_first_tp:], (eps + mean_PSTH_booted[:,fano_PSTH_first_tp:]))
        fano_PSTH_RF_SD = np.std(fano_PSTH_RF,axis=0)

        PSTH_RF = mean_PSTH_booted[:,fano_PSTH_first_tp:]/(count_window/1000)
        PSTH_RF_SD = np.std(PSTH_RF,axis=0)
        
        plt.figure(1,figsize=(1.335, 1.115))
        ax = plt.subplot(2,1,a+1)
        axb = ax.twinx()
        t = np.arange(-100,600,1)
        # plot fano-PSTH
        ax.fill_between(t,np.mean(fano_PSTH_RF,axis=0) - fano_PSTH_RF_SD, np.mean(fano_PSTH_RF,axis=0) + fano_PSTH_RF_SD,color='red',alpha=0.3)
        ax.plot(t,np.mean(fano_PSTH_RF,axis=0), 'k-')
        ax.set_ylim(0,12)
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
        axb.set_ylim(0,250)
        axb.spines['left'].set_color('red')
        axb.spines['top'].set_visible(False)
        axb.tick_params(axis='y',labelsize=8)
        #axb.set_ylabel('Firing-rate (Hz)')
        
        plt.figure(2)
        ax = plt.subplot(2,1,1)
        t = np.arange(50,450,1)
        # plot fano-PSTH
        ax.fill_between(t,np.mean(fano_PSTH_RF[:,150:550],axis=0) - fano_PSTH_RF_SD[150:550], np.mean(fano_PSTH_RF[:,150:550],axis=0) + fano_PSTH_RF_SD[150:550],color='red',alpha=0.3)
        ax.plot(t,np.mean(fano_PSTH_RF[:,150:550],axis=0), 'k-')
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
plt.savefig(fig_dir + 'F1_SG_PSTH_fanoPSTH.eps',bbox_inches='tight',pad_inches=0)        
plt.figure(2)
plt.savefig(fig_dir + 'F1_SG_fano-PSTH-zoomed.eps',bbox_inches='tight',pad_inches=0)
plt.figure(3)
plt.savefig(fig_dir + 'F1_SG_rasters.eps',bbox_inches='tight',pad_inches=0)

##
fano_E = 2 * np.std(fano_boot,axis=0)
FR_E   = 2 * np.std(FR_boot,axis=0)

plt.figure(4,figsize=(1.335, 1.115))
ax = plt.subplot(1,1,1)
axb = ax.twinx()
ax.errorbar(diamsa, fano, yerr=fano_E,fmt='ro-',markersize=4,mfc='None',lw=1)
axb.errorbar(diamsa, FR, yerr=FR_E,fmt='ko-',markersize=4,mfc='None',lw=1)
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
plt.savefig(fig_dir + 'F1_SG_ASFs.eps',bbox_inches='tight',pad_inches=0)
