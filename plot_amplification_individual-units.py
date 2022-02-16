import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
sys.path.append('C:/Users/lonurmin/Desktop/code/DataAnalysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols

import pdb

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

# analysis done between these timepoints
bsl_begin = 250

eps = 0.0000001

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)

diams = np.array(list(diams_data.keys()))
del(diams_data)

with open(S_dir + 'mean_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_mn_data = pkl.load(f)    
with open(S_dir + 'vari_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_vr_data = pkl.load(f)    




# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100
nboots = 3000

SG_perc_amplif = np.zeros((len(list(SG_mn_data.keys())),19))
SG_perc_quench = np.zeros((len(list(SG_mn_data.keys())),19))

G_perc_amplif = np.zeros((len(list(G_mn_data.keys())),19))
G_perc_quench = np.zeros((len(list(G_mn_data.keys())),19))

IG_perc_amplif = np.zeros((len(list(IG_mn_data.keys())),19))
IG_perc_quench = np.zeros((len(list(IG_mn_data.keys())),19))

t = np.arange(-150,600,1)



def process(data,mean_data,this_pdf,bsl_begin):

    for unit in list(mean_data.keys()):
        # loop diams
        mn_mtrx = mean_data[unit]
        
        Resp       = np.nan * np.ones((mn_mtrx.shape[0]))
        for stim in range(mn_mtrx.shape[0]):
            Resp[stim] = np.mean(mn_mtrx[stim,first_tp:last_tp])

        fig, ax = plt.subplots(3,7,figsize=(4,8))
        
        for stim in range(mn_mtrx.shape[0]):
            
            if mn_mtrx.shape[0] == 18:
                diam = diams[stim+1]
            else:
                diam = diams[stim]

            mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:],
                                                                                                    count_window,
                                                                                                    style='same',
                                                                                                    return_bootdstrs=True,
                                                                                                    nboots=nboots)

            fano_PSTH = vari_PSTH/(eps + mean_PSTH)
            fano_boot = np.divide(vari_PSTH_booted,np.add(eps,mean_PSTH_booted))
            PSTH_SE = np.std(mean_PSTH_booted,axis=0)
            fano_SE = np.std(fano_boot,axis=0)

            # raster
            dalib.rasters(np.squeeze(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:]), t, ax[0,stim],color='black')        
            ax[0,stim].set_title('diam = ' + str(diam))
            # PSTH
            ax[1,stim].fill_between(t,mean_PSTH-PSTH_SE,mean_PSTH+PSTH_SE,color='black',alpha=0.5)
            ax[1,stim].plot(t,mean_PSTH,color='black')
            # fano PSTH
            ax[2,stim].fill_between(t,fano_PSTH-fano_SE,fano_PSTH+fano_SE,color='red',alpha=0.5)
            ax[2,stim].plot(t,fano_PSTH,color='red')

        # -- plot results for RF size stimulus
        stim = np.argmax(Resp)
        # raster
        dalib.rasters(np.squeeze(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:]), t, ax[0,stim],color='black')        
        ax[0,stim].set_title('RF')
        # PSTH
        ax[1,stim].fill_between(t,mean_PSTH-PSTH_SE,mean_PSTH+PSTH_SE,color='black',alpha=0.5)
        ax[1,stim].plot(t,mean_PSTH,color='black')
        # fano PSTH
        ax[2,stim].fill_between(t,fano_PSTH-fano_SE,fano_PSTH+fano_SE,color='red',alpha=0.5)
        ax[2,stim].plot(t,fano_PSTH,color='red')    
        this_pdf.savefig()


