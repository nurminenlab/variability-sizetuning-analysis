import numpy as np
import matplotlib.pyplot as plt
import tables as tb
import itertools as it
import scipy.stats as sts
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/DataPreprocess')
import datapreprocesslib as dpl
os.sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis')
import data_analysislib as dalib
import itertools
#from sklearn import linear_model as lm
#from sklearn import preprocessing as prepro
from matplotlib.gridspec import GridSpec
import pandas as pd
from statsmodels.formula.api import ols
from scipy.optimize import curve_fit
#from lmfit import Model
import scipy as sc
import pickle as pkl

# open file with the data tables
results_root = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
file_name    = results_root+'collated_correlation_data_combinedsorting_400ms_MK-July2020.h5'
data_file    = tb.open_file(file_name,'r')

tables_list   = data_file.list_nodes('/data_group')

spkC_thr = 1
sup_thr  = 0.0
this_contrast = 50.0
that_contrast = 100.0
boot_num = int(1e3)
eps = np.finfo(float).eps

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
outfile = 'selectedData_lenient_400ms_macaque_July2020.pkl'

w_dir = os.getcwd()
# loop through penetrations and animals
a = 0

selectedData = {}
# counter for pairs
pp = 0
for data_table in tables_list:
    
    a +=1
    print('Now analyzing case ', a)
    #####

    contrast = data_table.col('contrast')
    contrast = np.squeeze(contrast[0,:,:])

    # data selection
    #####
    SNR      = data_table.col('SNR')[:,0]
    spkC_NoL = data_table.col('spkC_NoL')
    spkR_NoL = data_table.col('spkR_NoL')
    RF_NoL   = data_table.col('RF_NoL')
    diam     = data_table.col('diams')[0,:]
    depth    = data_table.col('depth')[:,0]
    baseline = data_table.col('baseLine')
    wavForms = data_table.col('waveForms')
    maxAmpC  = data_table.col('maxAmpC')
    spkDur   = data_table.col('spkDur')
    layer    = data_table.col('layer')
    animal   = data_table.col('animal')
    penetr   = data_table.col('penetration')

    # find units responding >= spike threshold, at 100% contrast
    spkC_NoL_mn100 = np.mean(spkC_NoL[:,:,contrast==that_contrast],axis=2)

    b100 = np.mean(baseline[:,:,contrast==that_contrast],axis=2)
    b100 = np.mean(np.asmatrix(b100),axis=1)
    b100 = np.repeat(b100,spkC_NoL_mn100.shape[1],axis=1)
        
    spkC_NoL_mn100 = spkC_NoL_mn100 - b100

    inds_tmp = np.max(spkC_NoL_mn100,axis=1) >= spkC_thr
    inds = np.zeros(inds_tmp.shape[0],dtype=np.bool)
    for i in range(inds_tmp.shape[0]):
        inds[i] = inds_tmp[i,0]

    
    spkC_NoL = spkC_NoL[inds,:,:]
    spkR_NoL = spkR_NoL[inds,:,:]
    SNR      = SNR[inds]
    RF_NoL   = RF_NoL[inds,:]
    depth    = depth[inds]
    baseline = baseline[inds,:,:]
    wavForms = wavForms[inds,:,:]
    maxAmpC  = maxAmpC[inds]
    spkDur   = spkDur[inds]
    layer    = layer[inds]
    animal   = animal[inds]
    penetr   = penetr[inds]
    
    # find units with suppression >= sup_thr
    inds = RF_NoL[:,2] >=sup_thr
    spkC_NoL = spkC_NoL[inds,:,:]
    spkR_NoL = spkR_NoL[inds,:,:]
    SNR      = SNR[inds]
    RF_NoL   = RF_NoL[inds,:]
    depth    = depth[inds]
    baseline = baseline[inds,:,:]
    wavForms = wavForms[inds,:,:]
    maxAmpC  = maxAmpC[inds]
    spkDur   = spkDur[inds]
    layer    = layer[inds]
    animal   = animal[inds]
    penetr   = penetr[inds]
    
    print('Numer of units: ', np.sum(inds))
        
    # iterate units
    for neuron in range(spkC_NoL.shape[0]):

        print('Now processing unit ', neuron)
        # loop contrast 
        C = np.unique(contrast)
        C = np.flipud(C)

        contrastResults = {}
        for pl_num, contr in enumerate(C):
            # containers
            fano_NoL = np.zeros((spkC_NoL.shape[1]))
            fano_bsl = np.zeros((spkC_NoL.shape[1]))
            
            # containers for bootstrap distributions
            fano_NoL_tmp = np.zeros((spkC_NoL.shape[1],boot_num))
            fano_bsl_tmp = np.zeros((baseline.shape[1],boot_num))

            # fano factor
            # the responses are multiplied to get rid of the scaling to spikes/s done in the preprocessing step,
            # as this would scale variance and mean differently and thus impact FANO
            fano_NoL = (np.var(spkC_NoL[neuron,:,contrast==contr].T,axis=1)) / (np.mean(spkC_NoL[neuron,:,contrast==contr].T,axis=1) + eps)
            fano_bsl = (np.var(baseline[neuron,:,contrast==contr].T,axis=1)) / (np.mean(baseline[neuron,:,contrast==contr].T,axis=1) + eps)

            bin_widths = np.array([25,50,100,200,400])
            fano_bins_NoL  = np.zeros((spkC_NoL.shape[1],bin_widths.shape[0]))
            for d in range(spkC_NoL.shape[1]):
                fano_bins_NoL[d,:] = dalib.fano_binwidths(spkR_NoL[neuron,d,contrast==contr,:],bin_widths)

            # bootstrap fano
            spkC_NoL_tmp = spkC_NoL[:,:,contrast==contr]
            baseline_tmp = baseline[:,:,contrast==contr]
            
            boot_inds = np.random.choice(np.sum(contrast==contr),(boot_num, np.sum(contrast==contr)))
            # the same exlanation for scaling as above
            for iter, b_inds in enumerate(boot_inds):
                fano_NoL_tmp[:,iter] = (np.var(spkC_NoL_tmp[neuron,:,b_inds].T,axis=1) + eps )/ (np.mean(spkC_NoL_tmp[neuron,:,b_inds].T,axis=1) + eps)
                fano_bsl_tmp[:,iter] = (np.var(baseline_tmp[neuron,:,b_inds].T,axis=1) + eps)/ (np.mean(baseline_tmp[neuron,:,b_inds].T,axis=1) + eps)

                
            # correlation data at each contrast
            contrastResults[contr] = {'spkC_NoL':spkC_NoL[neuron,:,contrast==contr],'baseline':baseline[neuron,:,contrast==contr],
                                      'fano_NoL':fano_NoL,'fano_bsl':fano_bsl,'boot_fano_NoL':fano_NoL_tmp,
                                      'boot_fano_bsl':fano_bsl_tmp,'fano_bins_NoL':fano_bins_NoL,
                                      'spkR_NoL':spkR_NoL[neuron,:,contrast==contr,:],}

        # collapsed results for a selected pair
        selectedData[pp] = contrastResults
        waves1 = wavForms[neuron,:,:]
        waves1 = waves1[~np.isnan(waves1[:,0]),:]
        selectedData[pp]['info'] = {'spikewidth1':spkDur[neuron],'SNR1':SNR[neuron],
                                    'waves1':waves1,'diam':diam,'layer':layer[neuron],
                                    'animal':animal[neuron],'penetr':penetr[neuron]}
        pp += 1
        
os.chdir(w_dir)
with open(F_dir + outfile,'wb') as f:
    pkl.dump(selectedData,f,pkl.HIGHEST_PROTOCOL)
