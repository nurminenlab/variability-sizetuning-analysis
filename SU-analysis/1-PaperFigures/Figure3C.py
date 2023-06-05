import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import scipy.stats as sts
import numpy as np
import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import warnings
warnings.filterwarnings('ignore')

save_figures = False

fig_dir  = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
dada_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/'

amplification_DF = pd.read_csv(dada_dir + 'amplification_DF_division.csv')
n_boots = 3000


# stimulus diameter relative to RF at maxquench / maxmplif 
RFnormed_maxamplif_diam = amplification_DF['maxamplif_diam']/amplification_DF['RFdiam']
amplification_DF.insert(loc=2,column='RFnormed_maxamplif_diam',value=RFnormed_maxamplif_diam)

# stimulus diameter at maxquench / maxmplif 
SG = amplification_DF.query('layer=="SG"')
G  = amplification_DF.query('layer=="G"')
IG = amplification_DF.query('layer=="IG"')

maxamplif_diam_RFnormed_bootstrap_SG = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_G  = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_IG = np.nan * np.ones(n_boots)

# Compute RF normed stimulus diameter at maxquench and maxamplif
# ----------------------------------------------------------------------------------------------------------------------
SG = SG[~np.isnan(SG['maxamplif_diam'].values)]
G = G[~np.isnan(G['maxamplif_diam'].values)]
IG = IG[~np.isnan(IG['maxamplif_diam'].values)]

# get indices to outliers
# RF normed stimulus diameter at maxamplification
SG['RFnormed_maxamplif_diam_outliers'] = SG['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(SG['RFnormed_maxamplif_diam'].median(),SG['RFnormed_maxamplif_diam'].mad()))
G['RFnormed_maxamplif_diam_outliers'] = G['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(G['RFnormed_maxamplif_diam'].median(),G['RFnormed_maxamplif_diam'].mad()))
IG['RFnormed_maxamplif_diam_outliers'] = IG['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(IG['RFnormed_maxamplif_diam'].median(),IG['RFnormed_maxamplif_diam'].mad()))

for i in range(n_boots):    
    
    maxamplif_diam_RFnormed_bootstrap_SG[i] = np.nanmedian(np.random.choice(SG['RFnormed_maxamplif_diam'][SG['RFnormed_maxamplif_diam_outliers']==False],
                                                            size=np.sum(~SG['RFnormed_maxamplif_diam_outliers']),replace=True))
    maxamplif_diam_RFnormed_bootstrap_G[i]  = np.nanmedian(np.random.choice(G['RFnormed_maxamplif_diam'][G['RFnormed_maxamplif_diam_outliers']==False],
                                                            size=np.sum(~G['RFnormed_maxamplif_diam_outliers']),replace=True))
    maxamplif_diam_RFnormed_bootstrap_IG[i] = np.nanmedian(np.random.choice(IG['RFnormed_maxamplif_diam'][IG['RFnormed_maxamplif_diam_outliers']==False],
                                                            size=np.sum(~IG['RFnormed_maxamplif_diam_outliers']),replace=True))


# RFnormed stimulus diameter @ maxquench

# RFnormed stimulus diameter @ maxamplif
# bootstrapped SEM
RFnormed_maxamplif_diam_SD = np.array([np.nanstd(maxamplif_diam_RFnormed_bootstrap_SG),
                                        np.nanstd(maxamplif_diam_RFnormed_bootstrap_G),
                                        np.nanstd(maxamplif_diam_RFnormed_bootstrap_IG)])
# median
RFnormed_maxamplif_diam = np.array([np.nanmedian(SG['RFnormed_maxamplif_diam'][SG['RFnormed_maxamplif_diam_outliers']==False]),
                                    np.nanmedian(G['RFnormed_maxamplif_diam'][G['RFnormed_maxamplif_diam_outliers']==False]),
                                    np.nanmedian(IG['RFnormed_maxamplif_diam'][IG['RFnormed_maxamplif_diam_outliers']==False])])

# Remove outliers, RF normalized amplification diameter
aSG_olRem_diam = pd.DataFrame(SG[SG['RFnormed_maxamplif_diam_outliers']==False])
aG_olRem_diam  = pd.DataFrame(G[G['RFnormed_maxamplif_diam_outliers']==False])
aIG_olRem_diam = pd.DataFrame(IG[IG['RFnormed_maxamplif_diam_outliers']==False])

# swarmplot RF normalized max quech and amplif stimulus diameters 
plt.figure()
SZZ = 2

frame = [aSG_olRem_diam, aG_olRem_diam, aIG_olRem_diam]
aQF = pd.concat(frame)

ax = plt.subplot(2,2,3)
sns.swarmplot(x='layer',y='RFnormed_maxamplif_diam',data=aQF,color='orange',size=SZZ)
ax.set_ylim(0.05,70)
ax.set_yscale('log')

ax = plt.subplot(2,2,4)
aQF.groupby('layer')['RFnormed_maxamplif_diam'].median().plot(kind='bar',yerr=RFnormed_maxamplif_diam_SD, color='white',ec='black')
ax.set_ylim(0.05,70)
ax.set_yscale('log')

if save_figures:
    plt.savefig(fig_dir+'Figure_3C-amp_quench_diameters.svg')
