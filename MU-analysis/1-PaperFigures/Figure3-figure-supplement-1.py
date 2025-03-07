import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.backends.backend_pdf import PdfPages

S_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/population-psths/'
F_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/2-PrecomputedAnalysis/GPFA-long-rasters/'
penetration  = 'MK374P2'

# load the data
with open(F_dir + 'layers_'+penetration+'.npy','rb') as f:
    layers = np.load(f)

with open(F_dir + penetration+'_stim_05.npy','rb') as f:
    binary_rasters = np.load(f)

layers = layers.astype(int)

# loop units
trial_start_time = 100

layer_colors = ['red','black','blue']
ax = plt.subplot(111)
for trial in range(0,binary_rasters.shape[2]):
    unit_idx = 0
    for unit in range(binary_rasters.shape[0]-1,0,-1):
        print(unit_idx)    
        spike_times = np.where(binary_rasters[unit,:,trial]==1)[0]
        ax.eventplot(spike_times - trial_start_time, color=layer_colors[layers[unit]-1], linewidths=0.5, lineoffsets=unit_idx+1)
        unit_idx += 1
    ax.set_yticks(np.arange(1,binary_rasters.shape[0]+2,1))
    ax.set_ylim(1,19.5)
    ax.set_yticklabels('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Unit')
    plt.savefig(S_dir + 'population-raster-'+penetration+'-trial_'+str(trial)+'.svg')
    ax.cla()
