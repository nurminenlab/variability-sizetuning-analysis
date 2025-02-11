import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
from scipy.optimize import basinhopping, curve_fit

import scipy.stats as sts

save_figures = True

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
                                'FR',
                                'penetration'])

G_params = pd.DataFrame(columns=['fano',
                                'bsl',                                
                                'diam',
                                'unit',
                                'bsl_FR',
                                'layer',
                                'FR',
                                'penetration'])

IG_params = pd.DataFrame(columns=['fano',
                                'bsl',                                
                                'diam',
                                'unit',
                                'bsl_FR',
                                'layer',
                                'FR',
                                'penetration'])

# loop SG units
indx   = 0
q_indx = 0
penetration_dict = {1:'MK366P1',2:'MK366P3',3:'MK366P8',4:'MK374P1',5:'MK374P2'}
SGpen = np.load(S_dir + 'SG-penetration.npy')
u_idx = 0
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

        para_tmp  = {'fano':fano,
                     'bsl':bsl_FF,
                     'bsl_FR':bsl_FR,
                     'diam':diam,
                     'layer':'SG',
                     'FR':FR,
                     'unit':unit,
                     'penetration':penetration_dict[SGpen[u_idx]]}
        
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        params     = params.append(tmp_df,sort=True)

        SG_tmp_df = pd.DataFrame(para_tmp, index=[q_indx])
        SG_params = SG_params.append(SG_tmp_df,sort=True)

        indx += 1
        q_indx += 1
    
    u_idx += 1
    
# loop G units
q_indx = 0
Gpen = np.load(S_dir + 'G-penetration.npy')
u_idx = 0
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

        
        para_tmp  = {'fano':fano,
                     'bsl':bsl_FF,
                     'bsl_FR':bsl_FR,
                     'diam':diam,
                     'layer':'G',
                     'FR':FR,
                     'unit':unit,
                     'penetration':penetration_dict[Gpen[u_idx]]}
        
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        params    = params.append(tmp_df,sort=True)
        
        G_tmp_df  = pd.DataFrame(para_tmp, index=[q_indx])
        G_params  = G_params.append(G_tmp_df,sort=True)

        indx += 1
        q_indx += 1

    u_idx += 1


# loop IG units
q_indx = 0
IGpen = np.load(S_dir + 'IG-penetration.npy')
u_idx = 0
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

        para_tmp  = {'fano':fano,
                     'bsl':bsl_FF,
                     'bsl_FR':bsl_FR,
                     'diam':diam,
                     'layer':'IG',
                     'FR':FR,
                     'unit':unit,
                     'penetration':penetration_dict[IGpen[u_idx]]}
        
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        params    = params.append(tmp_df,sort=True)

        IG_tmp_df  = pd.DataFrame(para_tmp, index=[q_indx])
        IG_params  = IG_params.append(IG_tmp_df,sort=True)

        indx += 1
        q_indx += 1

    u_idx += 1


ps = SG_params['penetration'].unique()
is_virgin = True
for penetr in ps:

    P = SG_params[SG_params['penetration'] == penetr]

    # supra-granular layer
    #
    sml = 0.2
    lrg = 50

    # loop units
    my_sizes = np.array([0.25, 0.5, 0.75, 
                        1.5, 2, 2.5, 3, 
                        4, 5, 6, 7, 8,
                        10, 15, 20, 30])

    units = P['unit'].unique()
    RFnormed_FR = np.nan * np.ones((len(units),len(my_sizes)+1))
    RFnormed_FF = np.nan * np.ones((len(units),len(my_sizes)+1))
    diams_tight = np.logspace(np.log10(P['diam'].values[0]),np.log10(P['diam'].values[-1]),1000)

    for ui, unit in enumerate(units):
        RFnormed = np.array([])
        # get unit params
        unit_params = P[P['unit']==unit]
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
            if si >= 3: # to skip the RF
                si = si + 1    
        
            RFnormed_FR[ui,si] = normed_FR[s]
            RFnormed_FF[ui,si] = FF[s]

    if is_virgin:
        RFnormed_FR_SG = np.nanmean(RFnormed_FR,axis=0)
        RFnormed_FF_SG = np.nanmean(RFnormed_FF,axis=0)
        is_virgin = False
    else:
        RFnormed_FR_SG = np.vstack((RFnormed_FR_SG,np.nanmean(RFnormed_FR,axis=0)))
        RFnormed_FF_SG = np.vstack((RFnormed_FF_SG,np.nanmean(RFnormed_FF,axis=0)))


my_sizes = np.insert(my_sizes,3,1)
plt.figure(1,figsize=(1.335, 1.115))
ax = plt.subplot(1,1,1)
for i in range(RFnormed_FR_SG.shape[0]):
    inds = ~np.isnan(RFnormed_FF_SG[i,:])    
    ax.semilogx(my_sizes[inds],RFnormed_FF_SG[i,inds],'-',lw=1)


if save_figures:
    plt.savefig(fig_dir + 'Figure2-Suppl-1-penetrations-SG.svg',bbox_inches='tight',pad_inches=0)

plt.close()


ps = G_params['penetration'].unique()
is_virgin = True
for penetr in ps:

    P = G_params[G_params['penetration'] == penetr]

    # supra-granular layer
    #
    sml = 0.2
    lrg = 50

    # loop units
    my_sizes = np.array([0.25, 0.5, 0.75, 
                        1.5, 2, 2.5, 3, 
                        4, 5, 6, 7, 8,
                        10, 15, 20, 30])

    units = P['unit'].unique()
    RFnormed_FR = np.nan * np.ones((len(units),len(my_sizes)+1))
    RFnormed_FF = np.nan * np.ones((len(units),len(my_sizes)+1))
    diams_tight = np.logspace(np.log10(P['diam'].values[0]),np.log10(P['diam'].values[-1]),1000)

    for ui, unit in enumerate(units):
        RFnormed = np.array([])
        # get unit params
        unit_params = P[P['unit']==unit]
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
            if si >= 3: # to skip the RF
                si = si + 1    
        
            RFnormed_FR[ui,si] = normed_FR[s]
            RFnormed_FF[ui,si] = FF[s]

    if is_virgin:
        RFnormed_FR_G = np.nanmean(RFnormed_FR,axis=0)
        RFnormed_FF_G = np.nanmean(RFnormed_FF,axis=0)
        is_virgin = False
    else:
        RFnormed_FR_G = np.vstack((RFnormed_FR_G,np.nanmean(RFnormed_FR,axis=0)))
        RFnormed_FF_G = np.vstack((RFnormed_FF_G,np.nanmean(RFnormed_FF,axis=0)))


my_sizes = np.insert(my_sizes,3,1)
plt.figure(1,figsize=(1.335, 1.115))
ax = plt.subplot(1,1,1)
for i in range(RFnormed_FR_G.shape[0]):
    inds = ~np.isnan(RFnormed_FF_G[i,:])    
    ax.semilogx(my_sizes[inds],RFnormed_FF_G[i,inds],'-',lw=1)


if save_figures:
    plt.savefig(fig_dir + 'Figure2-Suppl-1-penetrations-G.svg',bbox_inches='tight',pad_inches=0)


plt.close()

ps = IG_params['penetration'].unique()
is_virgin = True
for penetr in ps:

    P = IG_params[IG_params['penetration'] == penetr]

    # supra-granular layer
    #
    sml = 0.2
    lrg = 50

    # loop units
    my_sizes = np.array([0.25, 0.5, 0.75, 
                        1.5, 2, 2.5, 3, 
                        4, 5, 6, 7, 8,
                        10, 15, 20, 30])

    units = P['unit'].unique()
    RFnormed_FR = np.nan * np.ones((len(units),len(my_sizes)+1))
    RFnormed_FF = np.nan * np.ones((len(units),len(my_sizes)+1))
    diams_tight = np.logspace(np.log10(P['diam'].values[0]),np.log10(P['diam'].values[-1]),1000)

    for ui, unit in enumerate(units):
        RFnormed = np.array([])
        # get unit params
        unit_params = P[P['unit']==unit]
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
            if si >= 3: # to skip the RF
                si = si + 1    
        
            RFnormed_FR[ui,si] = normed_FR[s]
            RFnormed_FF[ui,si] = FF[s]

    if is_virgin:
        RFnormed_FR_IG = np.nanmean(RFnormed_FR,axis=0)
        RFnormed_FF_IG = np.nanmean(RFnormed_FF,axis=0)
        is_virgin = False
    else:
        RFnormed_FR_IG = np.vstack((RFnormed_FR_IG,np.nanmean(RFnormed_FR,axis=0)))
        RFnormed_FF_IG = np.vstack((RFnormed_FF_IG,np.nanmean(RFnormed_FF,axis=0)))


my_sizes = np.insert(my_sizes,3,1)
plt.figure(1,figsize=(1.335, 1.115))
ax = plt.subplot(1,1,1)
for i in range(RFnormed_FR_IG.shape[0]):
    inds = ~np.isnan(RFnormed_FF_IG[i,:])    
    ax.semilogx(my_sizes[inds],RFnormed_FF_IG[i,inds],'-',lw=1)


if save_figures:
    plt.savefig(fig_dir + 'Figure2-Suppl-1-penetrations-IG.svg',bbox_inches='tight',pad_inches=0)


plt.close()