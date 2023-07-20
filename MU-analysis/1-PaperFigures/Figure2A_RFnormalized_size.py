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
geo_mean = False

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'

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
    
with open(S_dir + 'mean_PSTHs_SG-MK-MU-newselection-Jun2023.pkl','rb') as f:
    SG_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_SG-MK-MU-newselection-Jun2023.pkl','rb') as f:
    SG_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_G-MK-MU-newselection-Jun2023.pkl','rb') as f:
    G_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_G-MK-MU-newselection-Jun2023.pkl','rb') as f:
    G_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_IG-MK-MU-newselection-Jun2023.pkl','rb') as f:
    IG_mn_data = pkl.load(f)    
with open(S_dir + 'vari_PSTHs_IG-MK-MU-newselection-Jun2023.pkl','rb') as f:
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
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / ( eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        FR   = np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0)    
        bsl_FF = np.mean(vr_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window] / (eps  + mn_mtrx[stim,bsl_begin:bsl_end][0:-1:count_window]))
        bsl_FR = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
                
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]

        
        para_tmp  = {'fano':fano,'bsl':bsl_FF,'bsl_FR':bsl_FR,'diam':diam,'layer':'SG','FR':FR,'unit':unit}
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

        para_tmp  = {'fano':fano,'bsl':bsl_FF,'bsl_FR':bsl_FR,'diam':diam,'layer':'SG','FR':FR,'unit':unit}
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        params    = params.append(tmp_df,sort=True)

        IG_tmp_df  = pd.DataFrame(para_tmp, index=[q_indx])
        IG_params  = IG_params.append(IG_tmp_df,sort=True)

        indx += 1
        q_indx += 1


SG_params['RFnormed_bslFR'] = SG_params['bsl_FR'] / SG_params['FR']
G_params['RFnormed_bslFR']  = G_params['bsl_FR'] / G_params['FR']
IG_params['RFnormed_bslFR'] = IG_params['bsl_FR'] / IG_params['FR']

# supra-granular layer
# loop units
my_sizes = np.array([0.25, 0.5, 0.75, 
                     1.5, 2, 2.5, 3, 
                     4, 5, 6, 7, 8,
                     10, 15, 20, 30])


units = SG_params['unit'].unique()
RFnormed_FR = np.nan * np.ones((len(units),len(my_sizes)+1))
RFnormed_FF = np.nan * np.ones((len(units),len(my_sizes)+1))
diams_tight = np.logspace(np.log10(SG_params['diam'].values[0]),np.log10(SG_params['diam'].values[-1]),1000)

for ui, unit in enumerate(units):
    RFnormed = np.array([])
    # get unit params
    unit_params = SG_params[SG_params['unit']==unit]
    # fit data
    try:
        popt,pcov = curve_fit(dalib.ROG,unit_params['diam'].values,unit_params['FR'].values,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (unit_params['diam'].values,unit_params['FR'].values)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        popt = res.x

    Rhat = dalib.ROG(diams_tight,*popt)
    RF   = diams_tight[np.argmax(Rhat)]
    RF   = unit_params['diam'].values[np.argmin(np.abs(unit_params['diam'].values - RF))]
    RFnormed = np.append(RFnormed,unit_params['diam'].values/RF)
    FR       = unit_params['FR'].values    
    FF       = unit_params['fano'].values    
    RFi = np.argmin(np.abs(RFnormed-1))
    normed_FR = FR/np.max(FR)
    FF = FF/FF[RFi]
    RFnormed_FR[ui,3] = 1
    RFnormed_FF[ui,3] = FF[RFi]
    RFnormed = np.delete(RFnormed,RFi)
    FR = np.delete(FR,RFi)
    FF = np.delete(FF,RFi)

    for s in range(len(RFnormed)):
        si = np.argmin(np.abs(my_sizes - RFnormed[s]))
        if si >= 3:
            si = si + 1    
      
        RFnormed_FR[ui,si] = normed_FR[s]
        RFnormed_FF[ui,si] = FF[s]
        


my_sizes = np.insert(my_sizes,3,1)

# ROG fit normalized spike-count data 
SG_normed_FR = np.nanmean(RFnormed_FR,axis=0)

if geo_mean:
    SG_normed_FF = np.exp(np.nanmean(np.log(RFnormed_FF),axis=0)) # per reviewer suggestion, we use geometric mean
else:
    SG_normed_FF = np.nanmean(RFnormed_FF,axis=0)

diams_tight = np.logspace(np.log10(my_sizes[0]),np.log10(my_sizes[-1]),1000)
try:
    popt,pcov = curve_fit(dalib.ROG,my_sizes[~np.isnan(SG_normed_FR)],SG_normed_FR,bounds=(0,np.inf),maxfev=100000)
except:
    args = (my_sizes[~np.isnan(SG_normed_FR)],SG_normed_FR)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x

Rhat = dalib.ROG(diams_tight,*popt)

# double ROG fit fano data 
args = (my_sizes[~np.isnan(SG_normed_FR)],SG_normed_FF)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)

Fhat = dalib.doubleROG(diams_tight,*res.x)

RFnormed_FF_divg = np.nan * np.ones(RFnormed_FF.shape)
for i in range(RFnormed_FF.shape[1]):
    RFnormed_FF_divg[:,i] = RFnormed_FF[:,i]/SG_normed_FF[i]

plt.figure()
#plt.figure(figsize=(1.335, 1.115))
ax = plt.subplot(1,1,1)
ax2 = ax.twinx()
ax.set_title('SG')
YERR = np.nanstd(RFnormed_FR,axis=0)/np.sqrt(np.sum(~np.isnan(RFnormed_FR),axis=0))
ax2.errorbar(my_sizes,SG_normed_FR,yerr=YERR,fmt='ko')
ax2.set_xscale('log')
ax2.set_ylabel('Normalized firing rate')

if geo_mean:
    YERR = np.exp(np.sqrt(np.nanmean(np.log(RFnormed_FF_divg)**2,axis=0)))/np.sqrt(RFnormed_FF_divg.shape[0])
else:
    YERR = np.nanstd(RFnormed_FF,axis=0)/np.sqrt(np.sum(~np.isnan(RFnormed_FF),axis=0))

ax.errorbar(my_sizes, SG_normed_FF,yerr=YERR,fmt='ro')
ax.set_xscale('log')
ax.set_ylabel('Fano factor')
ax.yaxis.label.set_color('red')
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',labelcolor='red')

ax2.plot(diams_tight,Rhat,'k-')
ax.plot(diams_tight,Fhat,'r-')

# granular layer
# loop units
my_sizes = np.array([0.25, 0.5, 0.75, 
                     1.5, 2, 2.5, 3, 
                     4, 5, 6, 7, 8,
                     10, 15, 20, 30])

units = G_params['unit'].unique()
RFnormed_FR = np.nan * np.ones((len(units),len(my_sizes)+1))
RFnormed_FF = np.nan * np.ones((len(units),len(my_sizes)+1))
diams_tight = np.logspace(np.log10(G_params['diam'].values[0]),np.log10(G_params['diam'].values[-1]),1000)

for ui, unit in enumerate(units):
    RFnormed = np.array([])
    # get unit params
    unit_params = G_params[G_params['unit']==unit]
    # fit data
    try:
        popt,pcov = curve_fit(dalib.ROG,unit_params['diam'].values,unit_params['FR'].values,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (unit_params['diam'].values,unit_params['FR'].values)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        popt = res.x

    Rhat = dalib.ROG(diams_tight,*popt)
    RF   = diams_tight[np.argmax(Rhat)]
    RF   = unit_params['diam'].values[np.argmin(np.abs(unit_params['diam'].values - RF))]
    RFnormed = np.append(RFnormed,unit_params['diam'].values/RF)
    FR       = unit_params['FR'].values    
    FF       = unit_params['fano'].values    
    RFi = np.argmin(np.abs(RFnormed-1))
    FF = FF/FF[RFi]
    normed_FR = FR/np.max(FR)
    RFnormed_FR[ui,3] = 1
    RFnormed_FF[ui,3] = FF[RFi]
    RFnormed = np.delete(RFnormed,RFi)
    FR = np.delete(FR,RFi)
    FF = np.delete(FF,RFi)

    for s in range(len(RFnormed)):
        si = np.argmin(np.abs(my_sizes - RFnormed[s]))
        if si >= 3:
            si = si + 1

        RFnormed_FR[ui,si] = normed_FR[s]
        RFnormed_FF[ui,si] = FF[s]

my_sizes = np.insert(my_sizes,3,1)

# ROG fit normalized spike-count data 
G_normed_FR = np.nanmean(RFnormed_FR,axis=0)
if geo_mean:
    G_normed_FF = np.exp(np.nanmean(np.log(RFnormed_FF),axis=0)) # per reviewer suggestion, we use geometric mean
else:
    G_normed_FF = np.nanmean(RFnormed_FF,axis=0)

diams_tight = np.logspace(np.log10(my_sizes[0]),np.log10(my_sizes[-1]),1000)
try:
    popt,pcov = curve_fit(dalib.ROG,my_sizes[~np.isnan(G_normed_FR)],G_normed_FR,bounds=(0,np.inf),maxfev=100000)
except:
    args = (my_sizes[~np.isnan(G_normed_FR)],G_normed_FR)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x

Rhat = dalib.ROG(diams_tight,*popt)

# double ROG fit fano data 
args = (my_sizes[~np.isnan(G_normed_FR)],G_normed_FF)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)

Fhat = dalib.doubleROG(diams_tight,*res.x)

RFnormed_FF_divg = np.nan * np.ones(RFnormed_FF.shape)
for i in range(RFnormed_FF.shape[1]):
    RFnormed_FF_divg[:,i] = RFnormed_FF[:,i]/G_normed_FF[i]

plt.figure()
#plt.figure(figsize=(1.335, 1.115))

ax = plt.subplot(1,1,1)
ax2 = ax.twinx()
ax.set_title('G')
YERR = np.nanstd(RFnormed_FR,axis=0)/np.sqrt(np.sum(~np.isnan(RFnormed_FR),axis=0))
ax2.errorbar(my_sizes,np.nanmean(RFnormed_FR,axis=0),yerr=YERR,fmt='ko')
ax2.set_xscale('log')
ax2.set_ylabel('Normalized firing rate')

if geo_mean:
    YERR = np.exp(np.sqrt(np.nanmean(np.log(RFnormed_FF_divg)**2,axis=0)))/np.sqrt(RFnormed_FF_divg.shape[0])
else:
    YERR = np.nanstd(RFnormed_FF,axis=0)/np.sqrt(RFnormed_FF.shape[0])
    
ax.errorbar(my_sizes,np.nanmean(RFnormed_FF,axis=0),yerr=YERR,fmt='ro')
ax.set_xscale('log')
ax.set_ylabel('Fano factor')
ax.yaxis.label.set_color('red')
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',labelcolor='red')

ax2.plot(diams_tight,Rhat,'k-')
ax.plot(diams_tight,Fhat,'r-')

# infra-granular layer
# loop units
my_sizes = np.array([0.25, 0.5, 0.75, 
                     1.5, 2, 2.5, 3, 
                     4, 5, 6, 7, 8, 9,
                     10, 15, 20, 25, 30, 40])

units = IG_params['unit'].unique()
RFnormed_FR = np.nan * np.ones((len(units),len(my_sizes)+1))
RFnormed_FF = np.nan * np.ones((len(units),len(my_sizes)+1))
diams_tight = np.logspace(np.log10(IG_params['diam'].values[0]),np.log10(IG_params['diam'].values[-1]),1000)

for ui, unit in enumerate(units):
    RFnormed = np.array([])
    # get unit params
    unit_params = IG_params[IG_params['unit']==unit]
    # fit data
    try:
        popt,pcov = curve_fit(dalib.ROG,unit_params['diam'].values,unit_params['FR'].values,bounds=(0,np.inf),maxfev=100000)
    except:
        args = (unit_params['diam'].values,unit_params['FR'].values)
        bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
        res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
        popt = res.x

    Rhat = dalib.ROG(diams_tight,*popt)
    RF   = diams_tight[np.argmax(Rhat)]
    RF   = unit_params['diam'].values[np.argmin(np.abs(unit_params['diam'].values - RF))]
    RFnormed = np.append(RFnormed,unit_params['diam'].values/RF)
    FR       = unit_params['FR'].values    
    FF       = unit_params['fano'].values    
    RFi = np.argmin(np.abs(RFnormed-1))
    FF = FF/FF[RFi]
    normed_FR = FR/np.max(FR)
    RFnormed_FR[ui,3] = 1
    RFnormed_FF[ui,3] = FF[RFi]
    RFnormed = np.delete(RFnormed,RFi)
    FR = np.delete(FR,RFi)
    FF = np.delete(FF,RFi)

    for s in range(len(RFnormed)):
        si = np.argmin(np.abs(my_sizes - RFnormed[s]))
        if si >= 3:
            si = si + 1
        
        
        RFnormed_FR[ui,si] = normed_FR[s]
        RFnormed_FF[ui,si] = FF[s]

my_sizes = np.insert(my_sizes,3,1)

# ROG fit normalized spike-count data 
IG_normed_FR = np.nanmean(RFnormed_FR,axis=0)
if geo_mean:
    IG_normed_FF = np.exp(np.nanmean(np.log(RFnormed_FF),axis=0)) # per reviewer suggestion, we use geometric mean
else:   
    IG_normed_FF = np.nanmean(RFnormed_FF,axis=0)

diams_tight = np.logspace(np.log10(my_sizes[0]),np.log10(my_sizes[-1]),1000)
try:
    popt,pcov = curve_fit(dalib.ROG,my_sizes[~np.isnan(IG_normed_FR)],IG_normed_FR,bounds=(0,np.inf),maxfev=100000)
except:
    args = (my_sizes[~np.isnan(IG_normed_FR)],IG_normed_FR)
    bnds = np.array([[0.0001,0.0001,0,0,0],[30,30,100,100,None]]).T
    res  = basinhopping(cost_response,np.ones(5),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234)
    popt = res.x

Rhat = dalib.ROG(diams_tight,*popt)

# double ROG fit fano data 
args = (my_sizes[~np.isnan(IG_normed_FR)],IG_normed_FF)
bnds = np.array([[0.0001,1,0.0001,0.0001,0.0001,0,0,0,0,0],[1,30,30,30,100,100,100,100,None,None]]).T
res = basinhopping(cost_fano,np.ones(10),minimizer_kwargs={'method': 'L-BFGS-B', 'args':args,'bounds':bnds},seed=1234,niter=1000)

Fhat = dalib.doubleROG(diams_tight,*res.x)

RFnormed_FF_divg = np.nan * np.ones(RFnormed_FF.shape)
for i in range(RFnormed_FF.shape[1]):
    RFnormed_FF_divg[:,i] = RFnormed_FF[:,i]/IG_normed_FF[i]

plt.figure()
#plt.figure(figsize=(1.335, 1.115))
ax = plt.subplot(1,1,1)
ax.set_title('IG')
ax2 = ax.twinx()
YERR = np.nanstd(RFnormed_FR,axis=0)/np.sqrt(np.sum(~np.isnan(RFnormed_FR),axis=0))
ax2.errorbar(my_sizes,np.nanmean(RFnormed_FR,axis=0),yerr=YERR,fmt='ko')
ax2.set_xscale('log')
ax2.set_ylabel('Normalized firing rate')

if geo_mean:
    YERR = np.exp(np.sqrt(np.nanmean(np.log(RFnormed_FF_divg)**2,axis=0)))/np.sqrt(RFnormed_FF_divg.shape[0])
else:
    YERR = np.nanstd(RFnormed_FF,axis=0)/np.sqrt(RFnormed_FF.shape[0])

ax.errorbar(my_sizes,np.nanmean(RFnormed_FF,axis=0),yerr=YERR,fmt='ro')
ax.set_xscale('log')
ax.set_ylabel('Fano factor')
ax.yaxis.label.set_color('red')
ax.tick_params(axis='y',color='red')
ax.spines['left'].set_color('red')
ax2.spines['left'].set_color('red')
ax.tick_params(axis='y',labelcolor='red')

ax2.plot(diams_tight,Rhat,'k-')
ax.plot(diams_tight,Fhat,'r-')