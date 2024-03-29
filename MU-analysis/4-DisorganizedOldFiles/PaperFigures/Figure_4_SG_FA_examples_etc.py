import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts
import glob
import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
from matplotlib.backends.backend_pdf import PdfPages

F_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/GPFA-long-rasters/'
S_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/'

penetrations = ['MK366P1','MK366P3','MK366P8','MK374P1','MK374P2']
save_figures = False

t = np.arange(-100,600,1)

pn = 3
files = np.array(glob.glob(F_dir + penetrations[pn] + '_stim*'))
layer_file = glob.glob(F_dir + 'layers_' + penetrations[pn] + '.npy')[0]
layers = np.load(layer_file)

# stimulus size loop
files = files[[0,3,18]]
for i,f in enumerate(files):
    binarized_raster = np.load(f)
    SG_raster = binarized_raster[layers == 1,:,:]
            
    fig1,axs = plt.subplots(binarized_raster.shape[2],1,sharex=True)
    for tr in range(binarized_raster.shape[2]):
        dalib.rasters(SG_raster[:,:,tr],t,axeli=axs[tr],lw=1.1)        
        axs[tr].spines['right'].set_visible(False)
        axs[tr].spines['left'].set_visible(False)        
        
        axs[tr].set_xticks([])
        axs[tr].set_yticks([])
        

    axs[tr].set_xticks([0,250,500])
    if save_figures:
        plt.savefig(S_dir + 'IntermediateFigures/' + penetrations[pn] + '_stim' + str(i) + '_SG_raster.eps')

for i,f in enumerate(files):
    
    binarized_raster = np.load(f)
    G_raster  = binarized_raster[layers == 2,:,:]
        
    fig1,axs = plt.subplots(binarized_raster.shape[2],1,sharex=True)
    for tr in range(binarized_raster.shape[2]):
        dalib.rasters(G_raster[:,:,tr],t,axeli=axs[tr],lw=1.1)        
        axs[tr].spines['right'].set_visible(False)
        axs[tr].spines['left'].set_visible(False)
        
        axs[tr].set_xticks([])
        axs[tr].set_yticks([])
        

    axs[tr].set_xticks([0,250,500])
    if save_figures:
        plt.savefig(S_dir + 'IntermediateFigures/' + penetrations[pn] + '_stim' + str(i) + '_G_raster.eps')

for i,f in enumerate(files):
    binarized_raster = np.load(f)
    IG_raster  = binarized_raster[layers == 3,:,:]
        
    fig1,axs = plt.subplots(binarized_raster.shape[2],1,sharex=True)
    for tr in range(binarized_raster.shape[2]):
        dalib.rasters(IG_raster[:,:,tr],t,axeli=axs[tr],lw=1.1)        
        axs[tr].spines['right'].set_visible(False)
        axs[tr].spines['left'].set_visible(False)
        
        axs[tr].set_xticks([])
        axs[tr].set_yticks([])
        

    axs[tr].set_xticks([0,250,500])
    if save_figures:
        plt.savefig(S_dir + 'IntermediateFigures/' + penetrations[pn] + '_stim' + str(i) + '_IG_raster.eps')
    