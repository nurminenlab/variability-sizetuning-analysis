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

S_dir   = 'c:/users/lonurmin/desktop/correlatedvariability/results/SU-preprocessed/'
fig_dir   = 'c:/users/lonurmin/desktop/correlatedvariability/results/SU-figures/'

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

# we clean up units without much fano factor tuning
# AA SG_units_to_remove = [1,7,14,51,53,58,80]
# AA IG_units_to_remove = [20,31,32,34,46,77,81]

SG_units_to_remove = [1,6,7,11,14,51,53,58,72,79,80]
IG_units_to_remove = [3,17,20,31,32,34,46,67,73,75,77,81,82]
# 
for unit in SG_units_to_remove:
    SG_mn_data.pop(unit)
    SG_vr_data.pop(unit)

for unit in IG_units_to_remove:
    IG_mn_data.pop(unit)
    IG_vr_data.pop(unit)

for unit in list(SG_mn_data.keys()):
    # loop diams
    mn_mtrx = SG_mn_data[unit]
    vr_mtrx = SG_vr_data[unit]

    for stim in range(mn_mtrx.shape[0]):        
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        FR   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)    
        bsl_FF = np.mean(vr_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window] / (eps + mn_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window]))
        bsl_FR = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
        if bsl_FF == 0:
            print('Unit:',unit)


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
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / ( eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        FR   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)    
        bsl_FF = np.mean(vr_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window] / (eps  + mn_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window]))
        bsl_FR = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
                
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]

        
        para_tmp  = {'fano':fano,'bsl':bsl_FF,'bsl_FR':bsl_FR,'diam':diam,'layer':'IG','FR':FR,'unit':unit}
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


# supragranular layer
plt.figure(figsize=(1.335, 1.115))
SG_params['FR'] = SG_params['FR'].apply(lambda x: x/(count_window/1000))

FR_size_tune = SG_params.groupby(['diam'], as_index=False)['FR'].mean()
FF_size_tune = SG_params.groupby(['diam'], as_index=False)['fano'].mean()

# fit data
try:
    popt,pcov = curve_fit(dalib.ROG,FR_size_tune['diam'].values,FR_size_tune['FR'].values,bounds=(0,np.inf),maxfev=100000)
except:
    args = (FR_size_tune['diam'].values,FR_size_tune['FR'].values)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x

args = (FF_size_tune['diam'].values,FF_size_tune['fano'].values)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
diams_tight = np.logspace(np.log10(FF_size_tune['diam'].values[0]),np.log10(FF_size_tune['diam'].values[-1]),1000)
Rhat = dalib.ROG(diams_tight,*popt)
Fhat = dalib.doubleROG(diams_tight,*res.x)


ax = plt.subplot(111)
ax2 = ax.twinx()

SG_bsl_FR = SG_params['bsl_FR'].apply(lambda x: x/(count_window/1000.0)).mean()
SG_bsl_FF = SG_params['bsl'].mean()

SG_params['log_fano'] = SG_params['fano'].apply(lambda x: np.log(x))
#SEM = SG_params.groupby(['diam'])['log_fano'].sem().apply(lambda x: np.exp(x))
SEM = SG_params.groupby(['diam'])['fano'].sem()
SG_params.groupby(['diam'])['fano'].mean().plot(yerr=SEM,ax=ax,kind='line',fmt='ro',markersize=4,mfc='None',lw=1)

ax.set_xlabel('')
ax.set_xscale('log')
ax.plot([SG_params['diam'].min(),SG_params['diam'].max()],[SG_bsl_FF,SG_bsl_FF],'r--')
# fits
ax.plot(diams_tight, Fhat, 'r-',lw=1)


SEM_FR = SG_params.groupby(['diam'])['FR'].sem()
SG_params.groupby(['diam'])['FR'].mean().plot(yerr=SEM_FR,ax=ax2,kind='line',fmt='ko',markersize=4,mfc='None',lw=1)
ax2.plot([SG_params['diam'].min(),SG_params['diam'].max()],[SG_bsl_FR,SG_bsl_FR],'k--')
ax2.plot(diams_tight, Rhat, 'k-',lw=1)
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red')
if save_figures:
    plt.savefig(fig_dir + 'F2A_SG_ASFs-average.svg',bbox_inches='tight',pad_inches=0)

FR = SG_params.groupby('diam')['FR'].mean()
D = SG_params['diam'].unique()[FR.argmax()]
D_max = SG_params['diam'].max()
FF_RF = SG_params[SG_params['diam'] == D]
FF_LAR = SG_params[SG_params['diam'] == D_max]
print(sts.ttest_rel(FF_RF['fano'].values,FF_LAR['fano'].values))

# granular layer
plt.figure(figsize=(1.335, 1.115))
G_params['FR'] = G_params['FR'].apply(lambda x: x/(count_window/1000))

FR_size_tune = G_params.groupby(['diam'], as_index=False)['FR'].mean()
FF_size_tune = G_params.groupby(['diam'], as_index=False)['fano'].mean()

# fit data
try:
    popt,pcov = curve_fit(dalib.ROG,FR_size_tune['diam'].values,FR_size_tune['FR'].values,bounds=(0,np.inf),maxfev=100000)
except:
    args = (FR_size_tune['diam'].values,FR_size_tune['FR'].values)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x

args = (FF_size_tune['diam'].values,FF_size_tune['fano'].values)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
diams_tight = np.logspace(np.log10(FF_size_tune['diam'].values[0]),np.log10(FF_size_tune['diam'].values[-1]),1000)
Rhat = dalib.ROG(diams_tight,*popt)
Fhat = dalib.doubleROG(diams_tight,*res.x)

ax = plt.subplot(111)
ax2 = ax.twinx()

G_bsl_FR = G_params['bsl_FR'].apply(lambda x: x/(count_window/1000.0)).mean()
G_bsl_FF = G_params['bsl'].mean()

SEM = G_params.groupby(['diam'])['fano'].sem()
G_params.groupby(['diam'])['fano'].mean().plot(yerr=SEM,ax=ax,kind='line',fmt='ro',markersize=4,mfc='None',lw=1)
ax.set_xlabel('')
ax.set_xscale('log')
ax.plot([G_params['diam'].min(),G_params['diam'].max()],[G_bsl_FF,G_bsl_FF],'r--')
# fits
ax.plot(diams_tight, Fhat, 'r-',lw=1)

SEM_FR = G_params.groupby(['diam'])['FR'].sem()
G_params.groupby(['diam'])['FR'].mean().plot(yerr=SEM_FR,ax=ax2,kind='line',fmt='ko',markersize=4,mfc='None',lw=1)
ax2.plot([G_params['diam'].min(),G_params['diam'].max()],[G_bsl_FR,G_bsl_FR],'k--')
ax2.plot(diams_tight, Rhat, 'k-',lw=1)
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red')
if save_figures:
    plt.savefig(fig_dir + 'F2A_G_ASFs-average.svg',bbox_inches='tight',pad_inches=0)

FR = G_params.groupby('diam')['FR'].mean()
D = G_params['diam'].unique()[FR.argmax()]
D_max = G_params['diam'].max()
FF_RF = G_params[G_params['diam'] == D]
FF_LAR = G_params[G_params['diam'] == D_max]

print(sts.ttest_rel(FF_RF['fano'].values,FF_LAR['fano'].values))

# IG layer
plt.figure(figsize=(1.335, 1.115))
IG_params['FR'] = IG_params['FR'].apply(lambda x: x/(count_window/1000))

FR_size_tune = IG_params.groupby(['diam'], as_index=False)['FR'].mean()
FF_size_tune = IG_params.groupby(['diam'], as_index=False)['fano'].mean()

# fit data
try:
    popt,pcov = curve_fit(dalib.ROG,FR_size_tune['diam'].values,FR_size_tune['FR'].values,bounds=(0,np.inf),maxfev=100000)
except:
    args = (FR_size_tune['diam'].values,FR_size_tune['FR'].values)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x

args = (FF_size_tune['diam'].values,FF_size_tune['fano'].values)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)
diams_tight = np.logspace(np.log10(FF_size_tune['diam'].values[0]),np.log10(FF_size_tune['diam'].values[-1]),1000)
Rhat = dalib.ROG(diams_tight,*popt)
Fhat = dalib.doubleROG(diams_tight,*res.x)

ax = plt.subplot(111)
ax2 = ax.twinx()

IG_bsl_FR = IG_params['bsl_FR'].apply(lambda x: x/(count_window/1000.0)).mean()
IG_bsl_FF = IG_params['bsl'].mean()

SEM = IG_params.groupby(['diam'])['fano'].sem()
IG_params.groupby(['diam'])['fano'].mean().plot(yerr=SEM,ax=ax,kind='line',fmt='ro',markersize=4,mfc='None',lw=1)
ax.set_xlabel('')
ax.set_xscale('log')
ax.plot([IG_params['diam'].min(),IG_params['diam'].max()],[IG_bsl_FF,IG_bsl_FF],'r--')
# fits
ax.plot(diams_tight, Fhat, 'r-',lw=1)

SEM_FR = IG_params.groupby(['diam'])['FR'].sem()
IG_params.groupby(['diam'])['FR'].mean().plot(yerr=SEM_FR,ax=ax2,kind='line',fmt='ko',markersize=4,mfc='None',lw=1)
ax2.plot([IG_params['diam'].min(),IG_params['diam'].max()],[IG_bsl_FR,IG_bsl_FR],'k--')
# fits
ax2.plot(diams_tight, Rhat, 'k-',lw=1)

ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',colors='red')
if save_figures:
    plt.savefig(fig_dir + 'F2A_IG_ASFs-average.svg',bbox_inches='tight',pad_inches=0)

FR = IG_params.groupby('diam')['FR'].mean()
D = IG_params['diam'].unique()[FR.argmax()]
D_max = IG_params['diam'].max()
FF_RF = IG_params[IG_params['diam'] == D]
FF_LAR = IG_params[IG_params['diam'] == D_max]
print(sts.ttest_rel(FF_RF['fano'].values,FF_LAR['fano'].values))

# just some garbage 