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
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

save_figures = False

# analysis done between these timepoints
bsl_begin = 250

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

t = np.arange(-150,600,1)

# comment/uncomment to plot different units
#units = [29]
#stims = np.array([1]) 

units = [103]
stims = np.array([1])

#units = [100]
#stims = np.array([4])

#units = [73]
#stims = np.array([1])

def process(data,mean_data,bsl_begin,t):
    anal_duration = 400
    first_tp  = 450
    last_tp   = first_tp + anal_duration

    for unit in units:
        
        

        fig1, ax1 = plt.subplots(1,2,figsize=(8,4))
        fig2, ax2 = plt.subplots(2,2,figsize=(8,4))

        for count, stim in enumerate(stims):
            
            mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin:],
                                                                                                    count_window=100,
                                                                                                    style='same',
                                                                                                    return_bootdstrs=True,
                                                                                                    nboots=nboots)

            fano_PSTH = vari_PSTH/(eps + mean_PSTH)
            fano_boot = np.divide(vari_PSTH_booted,np.add(eps,mean_PSTH_booted))
            PSTH_SE = np.std(mean_PSTH_booted/0.1,axis=0)
            fano_SE = np.std(fano_boot,axis=0)

            mean_PSTH = mean_PSTH/0.1
            # raster
            # removes an artifact created by the long convolution window
            # for consistency with the PSTH, the artifact is removed from the raster as well
            st_plot = 100
            en_plot = -50            
            plt.figure(fig1)
            dalib.rasters(np.squeeze(data[unit][cont]['spkR_NoL'][:,stim,bsl_begin+st_plot:en_plot]), t[st_plot:en_plot], ax1[count],color='black')        
            ax1[count].set_xticks([0,250,500])

            plt.figure(fig2)
            # PSTH            
            ax2[0,count].fill_between(t[st_plot:en_plot],mean_PSTH[st_plot:en_plot]-PSTH_SE[st_plot:en_plot],
                                      mean_PSTH[st_plot:en_plot]+PSTH_SE[st_plot:en_plot],color='gray')
            ax2[0,count].plot(t[st_plot:en_plot],mean_PSTH[st_plot:en_plot],color='black')
            ax2[0,count].plot([t[st_plot],t[en_plot]], [np.mean(mean_PSTH[st_plot:151]),np.mean(mean_PSTH[st_plot:151])],'k--')
            #ax2[0,count].set_ylim(0,200) 
            ax2[0,count].set_xticks([0,250,500])
            # fano PSTH
            ax2[1,count].fill_between(t[st_plot:],fano_PSTH[st_plot:]-fano_SE[st_plot:],fano_PSTH[st_plot:]+fano_SE[st_plot:],color='red')
            ax2[1,count].plot([t[st_plot],t[-1]], [np.mean(fano_PSTH[st_plot:115]),np.mean(fano_PSTH[st_plot:115])],'r--')
            ax2[1,count].plot(t[st_plot:],fano_PSTH[st_plot:],color=[0.5, 0, 0])
            #ax2[1,count].set_ylim(0,5) 
            ax2[1,count].set_xticks([0,250,500])
    
    return fig1,fig2, fano_PSTH

fig1,fig2,sh = process(data,IG_mn_data,bsl_begin,t)

plt.figure(fig1)
if save_figures:
    plt.savefig(fig_dir+'F3A-amplification_example-rasters'+str(units[0])+'.eps')
plt.figure(fig2)
if save_figures:
    plt.savefig(fig_dir+'F3A-amplification_example-PSTHs'+str(units[0])+'.svg')


