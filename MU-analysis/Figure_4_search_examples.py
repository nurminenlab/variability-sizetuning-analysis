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


S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
data_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/'
#penetrations = ['MK366P1','MK366P3','MK366P8','MK374P1','MK374P2']
penetrations = ['MK374P2']

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
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

for p in penetrations:
    SG_netvariance = np.load(data_dir+'netvariance_SG'+p+'.npy')
    bsl_SG_netvariance = np.load(data_dir+'bsl_netvariance_SG'+p+'.npy')
    if SG_netvariance.shape[1] == 18:
        diamsas = diams[1:]
    else:
        diamsas = diams
            
    plt.figure(1)
    ax = plt.subplot(2,2,1)
    for u in range(SG_netvariance.shape[0]):
        ax.plot(diamsas,SG_netvariance[u,:],'-',color='gray')

    ax.errorbar(diamsas,np.mean(SG_netvariance,axis=0),
                yerr=np.std(SG_netvariance,axis=0)/np.sqrt(SG_netvariance.shape[0]),
                fmt='ro')
    ax.set_xscale('log')

    # fit FA
    args = (diamsas,np.mean(SG_netvariance,axis=0))
    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
    SG_netvariance_popt = res.x
    diams_tight = np.logspace(np.log10(diamsas[0]),np.log10(diamsas[-1]),1000)
    #SG_Rhat = dalib.ROG(diams_tight,*SG_FR_popt)
    SG_Fhat = dalib.doubleROG(diams_tight,*SG_netvariance_popt)
    ax.plot(diams_tight,SG_Fhat,'r-')
    #ax.plot([diamsas[0],diamsas[-1]],[np.mean(bsl_SG_netvariance),np.mean(bsl_SG_netvariance)],'r--')

for p in penetrations:
    G_netvariance = np.load(data_dir+'netvariance_G'+p+'.npy')
    bsl_G_netvariance = np.load(data_dir+'bsl_netvariance_G'+p+'.npy')
    if G_netvariance.shape[1] == 18:
        diamsas = diams[1:]
    else:
        diamsas = diams
    
    ax2 = plt.subplot(2,2,2)
    for u in range(G_netvariance.shape[0]):
        ax2.plot(diamsas,G_netvariance[u,:],'-',color='gray')

    ax2.errorbar(diamsas,np.mean(G_netvariance,axis=0),
                yerr=np.std(G_netvariance,axis=0)/np.sqrt(G_netvariance.shape[0]),
                fmt='ro')
    ax2.set_xscale('log')

    # fit FA
    args = (diamsas,np.mean(G_netvariance,axis=0))
    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
    G_netvariance_popt = res.x
    diams_tight = np.logspace(np.log10(diamsas[0]),np.log10(diamsas[-1]),1000)
    #SG_Rhat = dalib.ROG(diams_tight,*SG_FR_popt)
    G_Fhat = dalib.doubleROG(diams_tight,*G_netvariance_popt)
    ax2.plot(diams_tight,G_Fhat,'r-')
    #ax2.plot([diamsas[0],diamsas[-1]],[np.mean(bsl_G_netvariance),np.mean(bsl_G_netvariance)],'r--')
    
for p in penetrations:
    IG_netvariance = np.load(data_dir+'netvariance_IG'+p+'.npy')
    bsl_IG_netvariance = np.load(data_dir+'bsl_netvariance_IG'+p+'.npy')
    if IG_netvariance.shape[1] == 18:
        diamsas = diams[1:]
    else:
        diamsas = diams
            
    ax3 = plt.subplot(2,2,3)
    for u in range(IG_netvariance.shape[0]):
        ax3.plot(diamsas,IG_netvariance[u,:],'-',color='gray')

    ax3.errorbar(diamsas,np.mean(IG_netvariance,axis=0),
                yerr=np.std(IG_netvariance,axis=0)/np.sqrt(IG_netvariance.shape[0]),
                fmt='ro')
    ax3.set_xscale('log')

    # fit FA
    args = (diamsas,np.mean(IG_netvariance,axis=0))
    bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
    res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
    IG_netvariance_popt = res.x
    diams_tight = np.logspace(np.log10(diamsas[0]),np.log10(diamsas[-1]),1000)
    #SG_Rhat = dalib.ROG(diams_tight,*SG_FR_popt)
    IG_Fhat = dalib.doubleROG(diams_tight,*IG_netvariance_popt)
    ax3.plot(diams_tight,IG_Fhat,'r-')
    #ax3.plot([diamsas[0],diamsas[-1]],[np.mean(bsl_IG_netvariance),np.mean(bsl_IG_netvariance)],'r--')
    plt.savefig(fig_dir + p + '_networkvariance-examples.svg')
    
    