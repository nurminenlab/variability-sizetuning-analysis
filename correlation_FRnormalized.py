# Import packages, load pre-computed data and do pre-processing
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as sts

sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
from scipy.optimize import basinhopping

def cost_fano(params,xdata,ydata):
    Rhat = dalib.doubleROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

FIG_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
params_df = pd.read_csv(F_dir + 'extracted_correlation_params.csv')
corrBSL = np.nanmean(params_df['gm_fit_correlation_BSL'].values)
params_df['utype'] = ['multi'] * len(params_df.index)

paramsas = params_df

# correlations for all pairs
with open(F_dir + 'correlations_all.pkl','rb') as f:
    correlations_all = pkl.load(f)

with open(F_dir + 'means_all.pkl','rb') as f:
    means_all = pkl.load(f)    
    
diams_all  = np.array(list(correlations_all.keys()))
rSC_mn_all = np.zeros(diams_all.shape[0])
rSC_SE_all = np.zeros(diams_all.shape[0])
gm_mn_all  = np.zeros(diams_all.shape[0])
gm_SE_all  = np.zeros(diams_all.shape[0])

for d in range(diams_all.shape[0]):
    FRnormed = correlations_all[diams_all[d]] / means_all[diams_all[d]]
    rSC_mn_all[d] = np.mean(FRnormed)
    rSC_SE_all[d] = np.std(FRnormed) / np.sqrt(FRnormed.shape[0])

    gm_mn_all[d] = np.mean(means_all[diams_all[d]])
    gm_SE_all[d] = np.std(means_all[diams_all[d]]) / np.sqrt(means_all[diams_all[d]].shape[0])
    
    
# correlations for supragranular layer units
with open(F_dir + 'correlations_SGSG.pkl','rb') as f:
    correlations_SGSG = pkl.load(f)
with open(F_dir + 'means_SGSG.pkl','rb') as f:
    means_SGSG = pkl.load(f)
    
diams_SGSG  = np.array(list(correlations_SGSG.keys()))
rSC_mn_SGSG = np.zeros(diams_SGSG.shape[0])
rSC_SE_SGSG = np.zeros(diams_SGSG.shape[0])
gm_mn_SGSG = np.zeros(diams_SGSG.shape[0])
gm_SE_SGSG = np.zeros(diams_SGSG.shape[0])

for d in range(diams_SGSG.shape[0]):
    FRnormed = correlations_SGSG[diams_all[d]] / means_SGSG[diams_all[d]]
    rSC_mn_SGSG[d] = np.mean(FRnormed)
    rSC_SE_SGSG[d] = np.std(FRnormed) / np.sqrt(FRnormed.shape[0])

    gm_mn_SGSG[d] = np.mean(means_SGSG[diams_SGSG[d]])
    gm_SE_SGSG[d] = np.std(means_SGSG[diams_SGSG[d]]) / np.sqrt(means_SGSG[diams_SGSG[d]].shape[0])
    
# correlations for infragranular units
with open(F_dir + 'correlations_IGIG.pkl','rb') as f:
    correlations_IGIG = pkl.load(f)
# correlations for narrow spiking units
with open(F_dir + 'means_IGIG.pkl','rb') as f:
    means_IGIG = pkl.load(f)
    
diams_IGIG  = np.array(list(correlations_IGIG.keys()))
rSC_mn_IGIG = np.zeros(diams_IGIG.shape[0])
rSC_SE_IGIG = np.zeros(diams_IGIG.shape[0])
gm_mn_IGIG = np.zeros(diams_IGIG.shape[0])
gm_SE_IGIG = np.zeros(diams_IGIG.shape[0])
for d in range(diams_IGIG.shape[0]):
    FRnormed = correlations_IGIG[diams_all[d]] / means_IGIG[diams_all[d]]
    rSC_mn_IGIG[d] = np.mean(FRnormed)
    rSC_SE_IGIG[d] = np.std(FRnormed) / np.sqrt(FRnormed.shape[0])

    gm_mn_IGIG[d] = np.mean(means_IGIG[diams_IGIG[d]])
    gm_SE_IGIG[d] = np.std(means_IGIG[diams_IGIG[d]]) / np.sqrt(means_IGIG[diams_IGIG[d]].shape[0])

# correlations for granular units
with open(F_dir + 'correlations_GG.pkl','rb') as f:
    correlations_GG = pkl.load(f)
# correlations for narrow spiking units
with open(F_dir + 'means_GG.pkl','rb') as f:
    means_GG = pkl.load(f)
    
diams_GG  = np.array(list(correlations_GG.keys()))
rSC_mn_GG = np.zeros(diams_GG.shape[0])
rSC_SE_GG = np.zeros(diams_GG.shape[0])
gm_mn_GG = np.zeros(diams_GG.shape[0])
gm_SE_GG = np.zeros(diams_GG.shape[0])
for d in range(diams_GG.shape[0]):
    FRnormed = correlations_GG[diams_all[d]] / means_GG[diams_all[d]]
    rSC_mn_GG[d] = np.mean(FRnormed)
    rSC_SE_GG[d] = np.std(FRnormed) / np.sqrt(FRnormed.shape[0])

    gm_mn_GG[d] = np.mean(means_GG[diams_GG[d]])
    gm_SE_GG[d] = np.std(means_GG[diams_GG[d]]) / np.sqrt(means_GG[diams_GG[d]].shape[0])


plt.subplot(2,2,1)
plt.errorbar(diams_all, rSC_mn_all, rSC_SE_all, fmt='ro-')
plt.xscale('log')
plt.title('All units')

plt.subplot(2,2,2)
plt.errorbar(diams_SGSG, rSC_mn_SGSG, rSC_SE_SGSG, fmt='ro-')
plt.xscale('log')
plt.title('Supragranular units')

plt.subplot(2,2,3)
plt.errorbar(diams_GG, rSC_mn_GG, rSC_SE_GG, fmt='ro-')
plt.xscale('log')
plt.title('Granular units')

plt.subplot(2,2,4)
plt.errorbar(diams_IGIG, rSC_mn_IGIG, rSC_SE_IGIG, fmt='ro-')
plt.xscale('log')
plt.title('Infragranular units')