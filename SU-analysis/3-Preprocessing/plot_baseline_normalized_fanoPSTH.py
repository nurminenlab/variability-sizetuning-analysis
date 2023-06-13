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
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages

#import pdb

save_figures = False

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
F_dir   = S_dir
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'
MUdatfile = 'selectedData_macaque_Jun2023.pkl'

# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 250
bsl_end   = bsl_begin + anal_duration

eps = 0.0001

zero_baseline_SG = []



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


count_window = 100
# containers for mean and fano PSTHs
SG_mean = np.ones((len(SG_mn_data),19,1000))*np.nan
SG_fano = np.ones((len(SG_mn_data),19,1000))*np.nan
G_mean = np.ones((len(G_mn_data),19,1000))*np.nan
G_fano = np.ones((len(G_mn_data),19,1000))*np.nan
IG_mean = np.ones((len(IG_mn_data),19,1000))*np.nan
IG_fano = np.ones((len(IG_mn_data),19,1000))*np.nan

# collect SG data
for unit_indx, unit in enumerate(list(SG_mn_data.keys())):
    # loop diams
    mn_mtrx = SG_mn_data[unit]
    vr_mtrx = SG_vr_data[unit]

    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus             
            SG_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:] / (eps + np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151]))
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)                
            SG_fano[unit_indx,stim+1,:] = FF / (eps + np.mean(FF[bsl_begin:bsl_begin + 151]))
        else:           
            SG_mean[unit_indx,stim,:] = mn_mtrx[stim,:] / (eps + np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151]))
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)            
            SG_fano[unit_indx,stim,:] = FF / (eps + np.mean(FF[bsl_begin:bsl_begin + 151]))

pdf = PdfPages(fig_dir + 'baseline_normalized_fanoPSTH_SG.pdf')
for i in range(SG_fano.shape[0]):
    plt.plot(SG_fano[i,5,:])
    pdf.savefig()
    plt.cla()

pdf.close()

# collect G data
#------------------------------------------------------------------------------
for unit_indx, unit in enumerate(list(G_mn_data.keys())):
    # loop diams
    mn_mtrx = G_mn_data[unit]
    vr_mtrx = G_vr_data[unit]
    
    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus 
            G_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            G_fano[unit_indx,stim+1,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])
        else:
            G_mean[unit_indx,stim,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            G_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])


pdf = PdfPages(fig_dir + 'baseline_normalized_fanoPSTH_G.pdf')
for i in range(G_fano.shape[0]):
    plt.plot(G_fano[i,5,:])
    pdf.savefig()
    plt.cla()

pdf.close()

# collect IG data
#------------------------------------------------------------------------------
for unit_indx, unit in enumerate(list(IG_mn_data.keys())):
    # loop diams
    mn_mtrx = IG_mn_data[unit]
    vr_mtrx = IG_vr_data[unit]
    
    for stim in range(mn_mtrx.shape[0]):
        if mn_mtrx.shape[0] == 18: # if there is no 0.1 deg stimulus 
            IG_mean[unit_indx,stim+1,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            IG_fano[unit_indx,stim+1,:] = FF/np.mean(FF[bsl_begin:bsl_begin + 151])
        else:
            IG_mean[unit_indx,stim,:] = mn_mtrx[stim,:] / np.mean(mn_mtrx[stim,bsl_begin:bsl_begin + 151])
            FF = vr_mtrx[stim,:] / (mn_mtrx[stim,:] + eps)
            IG_fano[unit_indx,stim,:] = FF / np.mean(FF[bsl_begin:bsl_begin + 151])

pdf = PdfPages(fig_dir + 'baseline_normalized_fanoPSTH_IG.pdf')
for i in range(IG_fano.shape[0]):
    plt.plot(IG_fano[i,5,:])
    pdf.savefig()
    plt.cla()
    
pdf.close()