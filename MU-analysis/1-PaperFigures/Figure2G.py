import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')

params['RFnormed_maxQuenchDiam'] = params['fit_fano_MIN_diam'] / params['RFdiam']

SG_df = params.query('layer == "LSG"')
G_df  = params.query('layer == "L4C"')
IG_df = params.query('layer == "LIG"')

G_median = sts.bootstrap((G_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian).standard_error
IG_median = sts.bootstrap((IG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian).standard_error
SG_median = sts.bootstrap((SG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian).standard_error

medians = [G_median,IG_median,SG_median]

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxQuenchDiam'].median().plot(kind='bar',yerr=medians,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxQuenchDiam',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2G.svg')


params_FANO = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')
params['RFdiam'] = params_FANO['RFdiam']

print('RF_normed_maxQuenchDiam medians')
params.groupby('layer')['RFnormed_maxQuenchDiam'].median()

print('RF_normed_maxQuenchDiam bootstrapper errors for medians')
print('SG: ', medians[2])
print('G: ',medians[0])
print('IG: ',medians[1])

# set the same median for each layer
SG_df['RFnormed_maxQuenched_zeroed'] = SG_df['RFnormed_maxQuenchDiam'] - SG_df['RFnormed_maxQuenchDiam'].median()
G_df['RFnormed_maxQuenched_zeroed']  = G_df['RFnormed_maxQuenchDiam'] - G_df['RFnormed_maxQuenchDiam'].median()
IG_df['RFnormed_maxQuenched_zeroed'] = IG_df['RFnormed_maxQuenchDiam'] - IG_df['RFnormed_maxQuenchDiam'].median()

SG_distr = sts.bootstrap((SG_df['RFnormed_maxQuenched_zeroed'].values,),np.median).bootstrap_distribution
G_distr  = sts.bootstrap((G_df['RFnormed_maxQuenched_zeroed'].values,),np.median).bootstrap_distribution
IG_distr = sts.bootstrap((IG_df['RFnormed_maxQuenched_zeroed'].values,),np.median).bootstrap_distribution

SG_G  = np.abs(SG_distr - G_distr)
SG_IG = np.abs(SG_distr - IG_distr)
G_IG  = np.abs(G_distr - IG_distr)

print('FFsuppression SG vs L4C: p-value')
print(np.sum(SG_G > np.abs(SG_df['RFnormed_maxQuenchDiam'].median() - G_df['RFnormed_maxQuenchDiam'].median())) / len(SG_G))

print('FFsuppression SG vs IG: p-value')
print(np.sum(SG_IG > np.abs(SG_df['RFnormed_maxQuenchDiam'].median() - IG_df['RFnormed_maxQuenchDiam'].median())) / len(SG_IG))

print('FFsuppression IG vs L4C: p-value')  
print(np.sum(SG_IG > np.abs(IG_df['RFnormed_maxQuenchDiam'].median() - G_df['RFnormed_maxQuenchDiam'].median())) / len(G_IG))
