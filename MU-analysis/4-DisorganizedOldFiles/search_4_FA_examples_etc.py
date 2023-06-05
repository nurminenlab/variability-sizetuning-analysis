import numpy as np
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
S_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'

penetrations = ['MK366P1','MK366P3','MK366P8','MK374P1','MK374P2']

t = np.arange(-100,600,1)
for p in penetrations:
    files = glob.glob(F_dir + p + '_stim*')
    layer_file = glob.glob(F_dir + 'layers_' + p + '.npy')[0]
    layers = np.load(layer_file)

    SGpdf = PdfPages(S_dir + p + '-SG.pdf')
    Gpdf = PdfPages(S_dir + p + '-G.pdf')
    IGpdf = PdfPages(S_dir + p + '-IG.pdf')

    # stimulus size loop
    for f in files:
        binarized_raster = np.load(f)
        SG_raster = binarized_raster[layers == 1,:,:]
        G_raster  = binarized_raster[layers == 2,:,:]
        IG_raster = binarized_raster[layers == 3,:,:]
        
        fig1,axs = plt.subplots(binarized_raster.shape[2],1,sharex=True)
        for tr in range(binarized_raster.shape[2]):
            dalib.rasters(SG_raster[:,:,tr],t,axeli=axs[tr])
        plt.axis('off')
        SGpdf.savefig()

        fig1,axs = plt.subplots(binarized_raster.shape[2],1,sharex=True)
        for tr in range(binarized_raster.shape[2]):
            dalib.rasters(G_raster[:,:,tr],t,axeli=axs[tr])
        plt.axis('off')
        Gpdf.savefig()

        fig1,axs = plt.subplots(binarized_raster.shape[2],1,sharex=True)
        for tr in range(binarized_raster.shape[2]):
            dalib.rasters(IG_raster[:,:,tr],t,axeli=axs[tr])
        plt.axis('off')
        IGpdf.savefig()
    
    SGpdf.close()
    Gpdf.close()
    IGpdf.close()   