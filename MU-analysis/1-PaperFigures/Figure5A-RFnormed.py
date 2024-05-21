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

# this way of computing the mean firing-rate functions is cumbersome but we do it this way for consistency with the other scripts
save_figures = False

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'

data_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/2-PrecomputedAnalysis/'

SG_netvariance = np.load(data_dir+'netvariance_all_SG.npy')
G_netvariance  = np.load(data_dir+'netvariance_all_G.npy')
IG_netvariance = np.load(data_dir+'netvariance_all_IG.npy')

bsl_SG_netvariance = np.load(data_dir+'bsl_netvariance_all_SG.npy')
bsl_G_netvariance  = np.load(data_dir+'bsl_netvariance_all_G.npy')
bsl_IG_netvariance = np.load(data_dir+'bsl_netvariance_all_IG.npy')


# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

count_window = 100

def cost_response(params,xdata,ydata):
    Rhat = dalib.ROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

def cost_fano(params,xdata,ydata):
    Rhat = dalib.doubleROG(xdata,*params)
    err  = np.sum(np.power(Rhat - ydata,2))
    return err

eps = 0.0000001

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

# param tables
params = pd.DataFrame(columns=['fano',
                                'bsl',                                
                                'diam',
                                'unit',
                                'bsl_FR',
                                'layer',
                                'FR'])

SG_params = pd.DataFrame(columns=['fano',
                                'bsl',                                
                                'diam',
                                'unit',
                                'bsl_FR',
                                'layer',
                                'FR'])

G_params = pd.DataFrame(columns=['fano',
                                'bsl',                                
                                'diam',
                                'unit',
                                'bsl_FR',
                                'layer',
                                'FR'])

IG_params = pd.DataFrame(columns=['fano',
                                'bsl',                                
                                'diam',
                                'unit',
                                'bsl_FR',
                                'layer',
                                'FR'])

# loop SG units
indx   = 0
q_indx = 0
for unit in list(SG_mn_data.keys()):
    # loop diams
    mn_mtrx = SG_mn_data[unit]
    vr_mtrx = SG_vr_data[unit]

    for stim in range(mn_mtrx.shape[0]):        
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        FR   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)    
        bsl_FF = np.mean(vr_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window] / (eps + mn_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window]))
        bsl_FR = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
                
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]

        para_tmp  = {'fano':fano,'bsl':bsl_FF,'bsl_FR':bsl_FR,'diam':diam,'layer':'SG','FR':FR,'unit':unit}
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        params     = params.append(tmp_df,sort=True)

        SG_tmp_df    = pd.DataFrame(para_tmp, index=[q_indx])
        SG_params  = SG_params.append(SG_tmp_df,sort=True)

        indx += 1
        q_indx += 1

# loop G units
q_indx = 0
for unit in list(G_mn_data.keys()):
    # loop diams
    mn_mtrx = G_mn_data[unit]
    vr_mtrx = G_vr_data[unit]

    for stim in range(mn_mtrx.shape[0]):
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        FR   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)    
        bsl_FF = np.mean(vr_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window] / (eps + mn_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window]))
        bsl_FR = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)

        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]

        para_tmp  = {'fano':fano,'bsl':bsl_FF,'bsl_FR':bsl_FR,'diam':diam,'layer':'G','FR':FR,'unit':unit}
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        params    = params.append(tmp_df,sort=True)

        G_tmp_df  = pd.DataFrame(para_tmp, index=[q_indx])
        G_params  = G_params.append(G_tmp_df,sort=True)

        indx += 1
        q_indx += 1


# loop IG units
q_indx = 0
for unit in list(IG_mn_data.keys()):
    # loop diams
    mn_mtrx = IG_mn_data[unit]
    vr_mtrx = IG_vr_data[unit]

    for stim in range(mn_mtrx.shape[0]):
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        FR   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)    
        bsl_FF = np.mean(vr_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window] / (eps + mn_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window]))
        bsl_FR = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)

        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]

        para_tmp  = {'fano':fano,'bsl':bsl_FF,'bsl_FR':bsl_FR,'diam':diam,'layer':'IG','FR':FR,'unit':unit}
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        params    = params.append(tmp_df,sort=True)

        IG_tmp_df  = pd.DataFrame(para_tmp, index=[q_indx])
        IG_params  = IG_params.append(IG_tmp_df,sort=True)

        indx += 1
        q_indx += 1

SG_params['FR'] = SG_params['FR'].apply(lambda x: x/(count_window/1000))
G_params['FR']  = G_params['FR'].apply(lambda x: x/(count_window/1000))
IG_params['FR'] = IG_params['FR'].apply(lambda x: x/(count_window/1000))

SG_size_tune = SG_params.groupby(['diam'], as_index=False)['FR'].mean()
G_size_tune  = G_params.groupby(['diam'], as_index=False)['FR'].mean()
IG_size_tune = IG_params.groupby(['diam'], as_index=False)['FR'].mean()

SG_netshare_tune = np.nanmean(SG_netvariance,axis=0)
G_netshare_tune  = np.nanmean(G_netvariance,axis=0)
IG_netshare_tune = np.nanmean(IG_netvariance,axis=0)

diams_tight = np.logspace(np.log10(SG_size_tune['diam'].values[0]),np.log10(SG_size_tune['diam'].values[-1]),1000)
# fit SG FR data
try:
    SG_FR_popt,pcov = curve_fit(dalib.ROG,SG_size_tune['diam'].values,SG_size_tune['FR'].values,bounds=(0,np.inf),maxfev=100000)
except:
    args = (SG_size_tune['diam'].values,SG_size_tune['FR'].values)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    SG_FR_popt = res.x

# fit SG netvariance data
args = (SG_size_tune['diam'].values,SG_netshare_tune)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
SG_netvariance_popt = res.x
SG_Rhat = dalib.ROG(diams_tight,*SG_FR_popt)
SG_Fhat = dalib.doubleROG(diams_tight,*SG_netvariance_popt)

# fit G FR data
try:
    G_FR_popt,pcov = curve_fit(dalib.ROG,G_size_tune['diam'].values,G_size_tune['FR'].values,bounds=(0,np.inf),maxfev=100000)
except:
    args = (G_size_tune['diam'].values,G_size_tune['FR'].values)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    G_FR_popt = res.x

# fit SG netvariance data
args = (G_size_tune['diam'].values,G_netshare_tune)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
G_netvariance_popt = res.x
G_Rhat = dalib.ROG(diams_tight,*G_FR_popt)
G_Fhat = dalib.doubleROG(diams_tight,*G_netvariance_popt)

# fit IG FR data
try:
    IG_FR_popt,pcov = curve_fit(dalib.ROG,IG_size_tune['diam'].values,IG_size_tune['FR'].values,bounds=(0,np.inf),maxfev=100000)
except:
    args = (IG_size_tune['diam'].values,IG_size_tune['FR'].values)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    IG_FR_popt = res.x

# fit IG netvariance data
args = (IG_size_tune['diam'].values,IG_netshare_tune)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
IG_netvariance_popt = res.x

IG_Rhat = dalib.ROG(diams_tight,*IG_FR_popt)
IG_Fhat = dalib.doubleROG(diams_tight,*IG_netvariance_popt)

ax = plt.subplot(111)
ax2 = ax.twinx()
SEM = SG_params.groupby(['diam'])['FR'].sem()
SG_params.groupby(['diam'])['FR'].mean().plot(yerr=SEM,ax=ax2,kind='line',fmt='ko',markersize=4,mfc='None',lw=1)
ax2.plot(diams_tight,SG_Rhat,'k-',lw=2)
ax2.plot([SG_params['diam'].min(),SG_params['diam'].max()],
        [SG_params['bsl_FR'].mean()/(count_window/1000),SG_params['bsl_FR'].mean()/(count_window/1000)],'k--',lw=1)
ax2.set_xscale('log')
ax2.set_ylabel('Firing rate (Hz)',color='k')

SG_netshare_tune_SEM = np.nanstd(SG_netvariance,axis=0)/np.sqrt(SG_netvariance.shape[0])
ax.errorbar(diams,SG_netshare_tune,yerr=SG_netshare_tune_SEM,fmt='ro',markersize=4,mfc='None',lw=1)
ax.plot(diams_tight,SG_Fhat,'r-',lw=2)
ax.plot([SG_params['diam'].min(),SG_params['diam'].max()],[np.nanmean(bsl_SG_netvariance),np.nanmean(bsl_SG_netvariance)],'r--',lw=1)
ax.set_xscale('log')
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red')
ax.set_ylabel('Shared variance (au)',color='red')
ax.set_title('SG')

if save_figures:
    plt.savefig(fig_dir+'Figure5A_SG.svg',bbox_inches='tight')

plt.figure()
ax = plt.subplot(111)
ax2 = ax.twinx()
SEM = G_params.groupby(['diam'])['FR'].sem()
G_params.groupby(['diam'])['FR'].mean().plot(yerr=SEM,ax=ax2,kind='line',fmt='ko',markersize=4,mfc='None',lw=1)
ax2.plot(diams_tight,G_Rhat,'k-',lw=2)
ax2.plot([G_params['diam'].min(),G_params['diam'].max()],
        [G_params['bsl_FR'].mean()/(count_window/1000),G_params['bsl_FR'].mean()/(count_window/1000)],'k--',lw=1)
ax2.set_xscale('log')
ax2.set_ylabel('Firing rate (Hz)',color='k')

G_netshare_tune_SEM = np.nanstd(G_netvariance,axis=0)/np.sqrt(G_netvariance.shape[0])
ax.errorbar(diams,G_netshare_tune,yerr=G_netshare_tune_SEM,fmt='ro',markersize=4,mfc='None',lw=1)
ax.plot(diams_tight,G_Fhat,'r-',lw=2)
ax.plot([G_params['diam'].min(),G_params['diam'].max()],[np.nanmean(bsl_G_netvariance),np.nanmean(bsl_G_netvariance)],'r--',lw=1)
ax.set_xscale('log')
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red')
ax.set_ylabel('Shared variance (au)',color='red')
ax.set_title('G')

if save_figures:
    plt.savefig(fig_dir+'Figure5A_G.svg',bbox_inches='tight')

plt.figure()
ax = plt.subplot(111)
ax2 = ax.twinx()
SEM = IG_params.groupby(['diam'])['FR'].sem()
IG_params.groupby(['diam'])['FR'].mean().plot(yerr=SEM,ax=ax2,kind='line',fmt='ko',markersize=4,mfc='None',lw=1)
ax2.plot([IG_params['diam'].min(),IG_params['diam'].max()],
        [IG_params['bsl_FR'].mean()/(count_window/1000),IG_params['bsl_FR'].mean()/(count_window/1000)],'k--',lw=1)
ax2.plot(diams_tight,IG_Rhat,'k-',lw=2)
ax2.set_xscale('log')
ax2.set_ylabel('Firing rate (Hz)',color='k')

IG_netshare_tune_SEM = np.nanstd(IG_netvariance,axis=0)/np.sqrt(IG_netvariance.shape[0])
ax.errorbar(diams,IG_netshare_tune,yerr=IG_netshare_tune_SEM,fmt='ro',markersize=4,mfc='None',lw=1)
ax.plot(diams_tight,IG_Fhat,'r-',lw=2)
ax.plot([IG_params['diam'].min(),IG_params['diam'].max()],[np.nanmean(bsl_IG_netvariance),np.nanmean(bsl_IG_netvariance)],'r--',lw=1)
ax2.set_xscale('log')
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red')
ax.tick_params(axis='y',colors='red')
ax.set_ylabel('Shared variance (au)',color='red')
ax.set_title('IG')

if save_figures:
    plt.savefig(fig_dir+'Figure5A_IG.svg',bbox_inches='tight')
