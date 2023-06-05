import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
#import pdb
import statsmodels.api as sm

import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
from scipy.optimize import basinhopping, curve_fit

import scipy.stats as sts

save_figures = False

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'

data_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/'

with open(S_dir + 'mean_PSTHs_G-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)


diams = np.array(list(diams_data.keys()))
del(diams_data)

SG_FA_covariance = np.load(data_dir+'FA_covariances_SG.npy')
G_FA_covariance  = np.load(data_dir+'FA_covariances_G.npy')
IG_FA_covariance = np.load(data_dir+'FA_covariances_IG.npy')

SG_data = np.nan*np.ones((SG_FA_covariance.shape[0],SG_FA_covariance.shape[2]))
G_data  = np.nan*np.ones((G_FA_covariance.shape[0],G_FA_covariance.shape[2]))
IG_data = np.nan*np.ones((IG_FA_covariance.shape[0],IG_FA_covariance.shape[2]))


for i in range(SG_FA_covariance.shape[2]):
    SG_data[:,i] = np.diag(SG_FA_covariance[:,:,i])
    G_data[:,i]  = np.diag(G_FA_covariance[:,:,i])
    IG_data[:,i] = np.diag(IG_FA_covariance[:,:,i])


plt.figure(1)
SGax = plt.subplot(111)
plt.figure(2)
Gax  = plt.subplot(111)
plt.figure(3)
IGax = plt.subplot(111)

for i in range(SG_data.shape[0]):
    SGax.plot(diams,SG_data[i,:],'-',color='grey')

for i in range(G_data.shape[0]):
    Gax.plot(diams,G_data[i,:],'-',color='grey')

for i in range(IG_data.shape[0]):
    IGax.plot(diams,IG_data[i,:],'-',color='grey')

SGax.errorbar(diams,np.mean(SG_data,axis=0),
                yerr=np.std(SG_data,axis=0)/np.sqrt(SG_data.shape[0]),
                fmt='o-',
                color='r',
                linewidth=2)

SGax.set_xscale('log')


Gax.errorbar(diams,np.mean(G_data,axis=0),
                yerr=np.std(G_data,axis=0)/np.sqrt(G_data.shape[0]),
                fmt='o-',
                color='r',
                linewidth=2)

Gax.set_xscale('log')


IGax.errorbar(diams,np.mean(IG_data,axis=0),
                yerr=np.std(IG_data,axis=0)/np.sqrt(IG_data.shape[0]),
                fmt='o-',
                color='r',
                linewidth=2)

IGax.set_xscale('log')

plt.figure(1)
if save_figures:
    plt.savefig(fig_dir+'SG_FA_covariance_examples.svg')

plt.figure(2)
if save_figures:
    plt.savefig(fig_dir+'G_FA_covariance_examples.svg')

plt.figure(3)
if save_figures:
    plt.savefig(fig_dir+'IG_FA_covariance_examples.svg')


print('SG shared variance mean 0.1: ',np.mean(SG_data[:,0],axis=0))
print('SG shared variance SE 0.1: ',np.std(SG_data[:,0],axis=0)/np.sqrt(SG_data.shape[0]))
print('SG shared variance mean 0.5: ',np.mean(SG_data[:,3],axis=0))
print('SG shared variance SE 0.5: ',np.std(SG_data[:,3],axis=0)/np.sqrt(SG_data.shape[0]))
print('SG shared variance mean 26: ',np.mean(SG_data[:,-1],axis=0))
print('SG shared variance SE 26: ',np.std(SG_data[:,-1],axis=0)/np.sqrt(SG_data.shape[0]))
print('\n')
print('####################')
print('\n')

print('G shared variance mean 0.1: ',np.mean(G_data[:,0],axis=0))
print('G shared variance se 0.1: ',np.std(G_data[:,0],axis=0)/np.sqrt(G_data.shape[0]))
print('G shared variance mean 0.5: ',np.mean(G_data[:,3],axis=0))
print('G shared variance se 0.5: ',np.std(G_data[:,3],axis=0)/np.sqrt(G_data.shape[0]))
print('G shared variance mean 26: ',np.mean(G_data[:,-1],axis=0))
print('G shared variance se 26: ',np.std(G_data[:,-1],axis=0)/np.sqrt(G_data.shape[0]))

print('\n')
print('####################')
print('\n')

print('IG shared variance mean 0.1: ',np.mean(IG_data[:,0],axis=0))
print('IG shared variance se 0.1: ',np.std(IG_data[:,0],axis=0)/np.sqrt(IG_data.shape[0]))
print('IG shared variance mean 0.5: ',np.mean(IG_data[:,3],axis=0))
print('IG shared variance se 0.5: ',np.std(IG_data[:,3],axis=0)/np.sqrt(IG_data.shape[0]))
print('IG shared variance mean 26: ',np.mean(IG_data[:,-1],axis=0))
print('IG shared variance se 26: ',np.std(IG_data[:,-1],axis=0)/np.sqrt(IG_data.shape[0]))