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

figures_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
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
    
with open(S_dir + 'mean_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_mn_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_mn_data = pkl.load(f)    


# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100
nboots = 3000

t = np.arange(-150,600,1)

def process(data,mean_data,bsl_begin,t,diams,units,amplification_diams,layer):
    anal_duration = 400
    first_tp  = 450
    last_tp   = first_tp + anal_duration

    for ui,unit in enumerate(units):
        # loop diams
        mn_mtrx = mean_data[unit]

        fig, ax = plt.subplots(3,1,figsize=(8,4))
                    
        stim = np.where(diams.round(1) == amplification_diams[ui].round(1))[0]
        diam = diams[stim]                                    
        if mn_mtrx.shape[0] == 18:
            mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim-1,bsl_begin:],
                                                                                                count_window=100,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=nboots)
        else:
            mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:],
                                                                                                count_window=100,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=nboots)

        fano_PSTH = vari_PSTH/(eps + mean_PSTH)
        fano_boot = np.divide(vari_PSTH_booted,np.add(eps,mean_PSTH_booted))
        PSTH_SE = np.std(mean_PSTH_booted,axis=0)
        fano_SE = np.std(fano_boot,axis=0)
            
        anipe = data[unit]['info']['animal'].decode('utf-8') + data[unit]['info']['penetr'].decode('utf-8')
        # raster
        dalib.rasters(np.squeeze(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:]), t, ax[0],color='black')        
        if stim == 0:
            ax[0].set_title(str(np.around(diam,1))+' '+anipe)
            ax[0].set_xticklabels([])
        else:
            ax[0].set_title(str(np.around(diam,1)))
            ax[0].set_xticklabels([])
            
        # PSTH
        count_window_sec = count_window/1000
        bsl = np.mean(mean_PSTH[:151]/count_window_sec)
        ax[1].fill_between(t,mean_PSTH/count_window_sec-PSTH_SE/count_window_sec,mean_PSTH/count_window_sec+PSTH_SE/count_window_sec,color='black',alpha=0.5)
        ax[1].plot(t,mean_PSTH/count_window_sec,color='black')
        ax[1].plot([t[0],t[-1]],[bsl,bsl],'--',color='black')
        ax[1].set_xticklabels([])
        # fano PSTH
        bsl = np.mean(fano_PSTH[:151])
        ax[2].fill_between(t,fano_PSTH-fano_SE,fano_PSTH+fano_SE,color='red',alpha=0.5)
        ax[2].plot(t,fano_PSTH,color='red')
        ax[2].plot([t[0],t[-1]],[bsl,bsl],'--',color='red')
        plt.savefig(figures_dir+layer+'_amplification_example_'+str(unit)+str(diam)+'.svg')


SG_units = np.array([0,1,25,49,95])
SG_diameters = np.array([0.2,0.2,0.2,0.2,1.0])

G_units = np.array([27])
G_diameters = np.array([0.2])

IG_units = np.array([12,29,78,79,101])
IG_diameters = np.array([0.2,0.4,0.2,0.1,0.2])

#process(data,SG_mn_data,bsl_begin,t,diams,SG_units,SG_diameters,'SG')
#process(data,G_mn_data,bsl_begin,t,diams,G_units,G_diameters,'G')
process(data,IG_mn_data,bsl_begin,t,diams,IG_units,IG_diameters,'IG')

