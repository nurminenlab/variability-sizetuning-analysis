import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params = pd.read_csv(F_dir + 'FA-params-Aug-2023.csv')
params_FANO = pd.read_csv(F_dir + 'extracted_params-Dec-2021.csv')
params['RFdiam'] = params_FANO['RFdiam']
del params_FANO

params['RFnormed_maxQuenchDiam'] = params['fit_FA_MIN_diam'] / params['RFdiam']

SG_df = params.query('layer == "SG"')
G_df  = params.query('layer == "G"')
IG_df = params.query('layer == "IG"')

errors = np.ones((2,3))

# granular 
errors[0,0]  = sts.bootstrap((G_df['RFnormed_maxQuenchDiam'].values,),np.median,confidence_level=0.68).confidence_interval[0]
errors[1,0]  = sts.bootstrap((G_df['RFnormed_maxQuenchDiam'].values,),np.median,confidence_level=0.68).confidence_interval[1]
# infragranular
errors[0,1]  = sts.bootstrap((IG_df['RFnormed_maxQuenchDiam'].values,),np.median,confidence_level=0.68).confidence_interval[0]
errors[1,1]  = sts.bootstrap((IG_df['RFnormed_maxQuenchDiam'].values,),np.median,confidence_level=0.68).confidence_interval[1]

# supragranular
errors[0,2]  = sts.bootstrap((SG_df['RFnormed_maxQuenchDiam'].values,),np.median,confidence_level=0.68).confidence_interval[0]
errors[1,2]  = sts.bootstrap((SG_df['RFnormed_maxQuenchDiam'].values,),np.median,confidence_level=0.68).confidence_interval[1]

SG_median_1 = SG_df['RFnormed_maxQuenchDiam'].values - SG_df['RFnormed_maxQuenchDiam'].median() + 1
G_median_1  = G_df['RFnormed_maxQuenchDiam'].values - G_df['RFnormed_maxQuenchDiam'].median() + 1
IG_median_1 = IG_df['RFnormed_maxQuenchDiam'].values - IG_df['RFnormed_maxQuenchDiam'].median() + 1

SG_median_1_boot = sts.bootstrap((SG_median_1,),np.median).bootstrap_distribution
G_median_1_boot  = sts.bootstrap((G_median_1,),np.median).bootstrap_distribution
IG_median_1_boot = sts.bootstrap((IG_median_1,),np.median).bootstrap_distribution

print('SG p:',np.sum(SG_median_1_boot >= SG_df['RFnormed_maxQuenchDiam'].median()) / len(SG_median_1_boot))
print('G p:', np.sum(G_median_1_boot >= G_df['RFnormed_maxQuenchDiam'].median()) / len(G_median_1_boot))
print('IG p:', np.sum(IG_median_1_boot >= IG_df['RFnormed_maxQuenchDiam'].median()) / len(IG_median_1_boot))

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxQuenchDiam'].median().plot(kind='bar',yerr=errors,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxQuenchDiam',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F5E.svg')

print(params.groupby('layer')['RFnormed_maxQuenchDiam'].median())