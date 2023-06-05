# this scripts reorganizes binned spike data so that it works plug and play with gpfa estimator scripts 
import sys
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import pdb

anal_contrast = 100.0
# analysis done between these timepoints
anal_duration = 400
first_tp      = 50
last_tp       = first_tp + anal_duration
bsl_begin     = 50
bsl_end       = bsl_begin + anal_duration

F_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
S_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/GPFA-long-rasters/'

SI_crit = 0.05
    
MUfile  = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

with open(F_dir + MUfile,'rb') as f_MU:
    MUdata = pkl.load(f_MU)

# loop multiunits and gather penetration and animal info
animal_penetr = list()
for unit in range(len(MUdata)):
    animal_penetr.append(MUdata[unit]['info']['animal'].decode('utf-8') + MUdata[unit]['info']['penetr'].decode('utf-8'))

animal_penetr = np.array(animal_penetr)
for penetr in np.unique(animal_penetr):
    # find responsive units in a penetration 
    indx = np.where(animal_penetr == penetr)[0]
    # init output matrix
    selected_units = np.empty(0)
    layers         = np.empty(0)
    for unit in indx:
        if anal_contrast in MUdata[unit].keys() and dalib.select_data(MUdata[unit][anal_contrast]['spkC_NoL'].T,MUdata[unit][anal_contrast]['baseline'],3):
            Y = np.mean(MUdata[unit][anal_contrast]['spkC_NoL'].T,axis=1)
            SI = (np.max(Y) - Y[-1]) / np.max(Y)

            # perform quick anova to see if tuned
            dd = MUdata[unit][100.0]['spkC_NoL']
            dd2 = np.ones((dd.shape[0] * dd.shape[1],2)) * np.nan
            for i in range(dd.shape[1]):
                dd2[0+(i * dd.shape[0]):dd.shape[0] + (i * dd.shape[0]),0] = dd[:,i]
                dd2[0+(i * dd.shape[0]):dd.shape[0] + (i * dd.shape[0]),1] = i*np.ones(dd.shape[0])
                                                                             
            df = pd.DataFrame(data=dd2, columns=['FR','D'])
            lm = ols('FR ~ C(D)',data=df).fit()
            table = sm.stats.anova_lm(lm,typ=1)
            tuned = table['PR(>F)']['C(D)'] < 0.05
            
            if SI > SI_crit and tuned:
                selected_units = np.append(selected_units, unit)
                if MUdata[unit]['info']['layer'].decode('utf-8') == 'LSG':
                    L = 1
                elif MUdata[unit]['info']['layer'].decode('utf-8') == 'L4C':
                    L = 2
                else:
                    L = 3

                layers = np.append(layers, L)
                
                        
    if selected_units.shape[0] is not 0:
        np.save(S_dir+'unit_numbers_'+penetr+'.npy',selected_units)
        np.save(S_dir+'layers_'+penetr+'.npy',layers)
        
        # initialize matrix for spike counts and fill it 
        # nunits x anal_duration timepoints x ntrials
        spk_mtrx = np.zeros((selected_units.shape[0],anal_duration,MUdata[selected_units[0]][anal_contrast]['spkR_NoL'].shape[0]))
        bsl_mtrx = np.zeros((selected_units.shape[0],anal_duration,MUdata[selected_units[0]][anal_contrast]['spkR_NoL'].shape[0]))
        for ss_stim in range(MUdata[selected_units[0]][anal_contrast]['spkR_NoL'].shape[1]):
            for unit in range(selected_units.shape[0]):
                for tr in range(MUdata[selected_units[0]][anal_contrast]['spkR_NoL'].shape[0]):
                    spk_mtrx[unit,:,tr] = MUdata[selected_units[unit]][anal_contrast]['spkR_NoL'][tr,ss_stim,first_tp:last_tp]
                    bsl_mtrx[unit,:,tr] = MUdata[selected_units[unit]][anal_contrast]['spkR_NoL'][tr,ss_stim,bsl_begin:bsl_end]
                    
            # now write matrix using 
            #np.save(S_dir+str(penetr)+'_stim_'+str(ss_stim).zfill(2),spk_mtrx)
            np.save(S_dir+str(penetr)+'_bsl_'+str(ss_stim).zfill(2),bsl_mtrx)

