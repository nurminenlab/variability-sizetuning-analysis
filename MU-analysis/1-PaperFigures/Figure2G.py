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

G_median = sts.bootstrap((G_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian,confidence_level=0.99).confidence_interval
IG_median = sts.bootstrap((IG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian,confidence_level=0.99).confidence_interval
SG_median = sts.bootstrap((SG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian,confidence_level=0.99).confidence_interval

CIs = np.nan * np.ones((2,3))
medians = np.nan * np.ones((2,3))
CIs[0,:] = np.array([G_median.low,IG_median.low,SG_median.low])
CIs[1,:] = np.array([G_median.high,IG_median.high,SG_median.high])

for i in range(medians.shape[0]):
    medians[i,:] = np.abs(CIs[i,:]- np.array([G_df['RFnormed_maxQuenchDiam'].median(),
                                                  IG_df['RFnormed_maxQuenchDiam'].median(),
                                                  SG_df['RFnormed_maxQuenchDiam'].median()]))

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxQuenchDiam'].median().plot(kind='bar',yerr=medians,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxQuenchDiam',hue='anipe',data=params,ax=ax,size=3)
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2G.svg')

print('RF_normed_maxQuenchDiam medians')
print(params.groupby('layer')['RFnormed_maxQuenchDiam'].median())

print('\nRF_normed_maxQuenchDiam bootstrapper errors for medians')
print('RF_normed_maxQuenchDiam bootstrapped CI for medians')
print('Low: G, IG, SG ', [G_median.low,IG_median.low,SG_median.low])
print('High: G, IG, SG ', [G_median.high,IG_median.high,SG_median.high])


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

""" print('\nFFsuppression SG vs L4C: p-value')
print(np.sum(SG_G > np.abs(SG_df['RFnormed_maxQuenchDiam'].median() - G_df['RFnormed_maxQuenchDiam'].median())) / len(SG_G))

print('FFsuppression SG vs IG: p-value')
print(np.sum(SG_IG > np.abs(SG_df['RFnormed_maxQuenchDiam'].median() - IG_df['RFnormed_maxQuenchDiam'].median())) / len(SG_IG))

print('FFsuppression IG vs L4C: p-value')  
print(np.sum(SG_IG > np.abs(IG_df['RFnormed_maxQuenchDiam'].median() - G_df['RFnormed_maxQuenchDiam'].median())) / len(G_IG)) """



