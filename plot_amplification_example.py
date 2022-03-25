import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.backends.backend_pdf import PdfPages

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

# analysis done between these timepoints
bsl_begin = 100

eps = 0.0000001

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)

diams = np.array(list(diams_data.keys()))
del(diams_data)
    
with open(S_dir + 'mean_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_mn_data = pkl.load(f)

# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100
nboots = 3000

t = np.arange(-300,600,1)

def process(data,mean_data,this_pdf,bsl_begin,t,diams):
    anal_duration = 400
    first_tp  = 450
    last_tp   = first_tp + anal_duration

    for unit in [24]:
        # loop diams
        mn_mtrx = mean_data[unit]
        
        Resp       = np.nan * np.ones((mn_mtrx.shape[0]))
        for stim in range(mn_mtrx.shape[0]):
            Resp[stim] = np.mean(mn_mtrx[stim,first_tp:last_tp])

        fig, ax = plt.subplots(3,2,figsize=(8,4))
        
        for count, stim in np.array([0,np.argmax(Resp)]):
            
            if mn_mtrx.shape[0] == 18:
                diam = diams[stim+1]
            else:
                diam = diams[stim]

            mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:],
                                                                                                    count_window=100,
                                                                                                    style='same',
                                                                                                    return_bootdstrs=True,
                                                                                                    nboots=nboots)

            fano_PSTH = vari_PSTH/(eps + mean_PSTH)
            fano_boot = np.divide(vari_PSTH_booted,np.add(eps,mean_PSTH_booted))
            PSTH_SE = np.std(mean_PSTH_booted,axis=0)
            fano_SE = np.std(fano_boot,axis=0)

            # raster
            dalib.rasters(np.squeeze(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:]), t, ax[0,count],color='black')        
            ax[0,count].set_title(str(np.around(diam,1)))
            # PSTH
            ax[1,count].fill_between(t,mean_PSTH-PSTH_SE,mean_PSTH+PSTH_SE,color='black',alpha=0.5)
            ax[1,count].plot(t,mean_PSTH,color='black')
            # fano PSTH
            ax[2,count].fill_between(t,fano_PSTH-fano_SE,fano_PSTH+fano_SE,color='red',alpha=0.5)
            ax[2,count].plot(t,fano_PSTH,color='red')

process(data,IG_mn_data,bsl_begin,t,diams)
plt.savefig(S_dir+'amplification_example.svg')

