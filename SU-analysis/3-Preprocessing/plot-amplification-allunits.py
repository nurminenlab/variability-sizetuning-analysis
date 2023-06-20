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
from matplotlib.backends.backend_pdf import PdfPages
#import pdb

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
save_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/'
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'

MUdatfile = 'selectedData_macaque_Jun2023.pkl'

SG_pdf = PdfPages(fig_dir + 'SG_fano_PSTHs.pdf')
IG_pdf = PdfPages(fig_dir + 'IG_fano_PSTHs.pdf')

# analysis done between these timepoints
anal_duration = 300
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

eps = 0.0000001

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)

diams = np.array(list(diams_data.keys()))
del(diams_data)

with open(S_dir + 'mean_PSTHs_SG-MK-SU-Jun2023.pkl','rb') as f:
    SG_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_SG-MK-SU-Jun2023.pkl','rb') as f:
    SG_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_G-MK-SU-Jun2023.pkl','rb') as f:
    G_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_G-MK-SU-Jun2023.pkl','rb') as f:
    G_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_IG-MK-SU-Jun2023.pkl','rb') as f:
    IG_mn_data = pkl.load(f)    
with open(S_dir + 'vari_PSTHs_IG-MK-SU-Jun2023.pkl','rb') as f:
    IG_vr_data = pkl.load(f)    


# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100
nboots = 3000
t = np.arange(-280,600,1)

for unit_indx, unit in enumerate(list(SG_mn_data.keys())):
    # loop diams
    mn_mtrx = SG_mn_data[unit]
    vr_mtrx = SG_vr_data[unit]

    delta_fano = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl        = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl_FR     = np.nan * np.ones((mn_mtrx.shape[0]))
    signi_all  = np.nan * np.ones((mn_mtrx.shape[0]),dtype=object)
    Resp       = np.nan * np.ones((mn_mtrx.shape[0]))
    for stim in range(mn_mtrx.shape[0]):
        Resp[stim] = np.mean(mn_mtrx[stim,first_tp:last_tp])
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end] / (eps + mn_mtrx[stim,bsl_begin:bsl_end]))
        bsl_FR[stim] = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
        delta_fano[stim] = fano - bsl[stim]
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]
            

        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=nboots)

        fano = vari_PSTH[bsl_begin:] / (eps + mean_PSTH[bsl_begin:])
        fano_boot = vari_PSTH_booted[:,bsl_begin:] / (eps + mean_PSTH_booted[:,bsl_begin:])
        fano_SE = np.std(fano_boot,axis=0)

        ax = plt.subplot(4,5,stim+1)
        ax.fill_between(t,fano-fano_SE,fano+fano_SE,color='red')
        ax.plot([0,0],[0,np.max(fano)],'k--')
    
    SG_pdf.savefig()
    plt.clf()

for unit_indx, unit in enumerate(list(IG_mn_data.keys())):
    # loop diams
    mn_mtrx = IG_mn_data[unit]
    vr_mtrx = IG_vr_data[unit]

    delta_fano = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl        = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl_FR     = np.nan * np.ones((mn_mtrx.shape[0]))
    signi_all  = np.nan * np.ones((mn_mtrx.shape[0]),dtype=object)
    Resp       = np.nan * np.ones((mn_mtrx.shape[0]))
    for stim in range(mn_mtrx.shape[0]):
        Resp[stim] = np.mean(mn_mtrx[stim,first_tp:last_tp])
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end] / (eps + mn_mtrx[stim,bsl_begin:bsl_end]))
        bsl_FR[stim] = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
        delta_fano[stim] = fano - bsl[stim]
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]
            

        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=nboots)

        fano = vari_PSTH[bsl_begin:] / (eps + mean_PSTH[bsl_begin:])
        fano_boot = vari_PSTH_booted[:,bsl_begin:] / (eps + mean_PSTH_booted[:,bsl_begin:])
        fano_SE = np.std(fano_boot,axis=0)

        ax = plt.subplot(4,5,stim+1)
        ax.fill_between(t,fano-fano_SE,fano+fano_SE,color='red')
        ax.plot([0,0],[0,np.max(fano)],'k--')
    
    IG_pdf.savefig()
    plt.clf()

SG_pdf.close()
IG_pdf.close()

