import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
params_df = pd.read_csv(F_dir + 'extracted_params-Dec-2021.csv')

params = params_df[['layer','fit_fano_SML','fit_fano_RF','fit_fano_SUR','fit_fano_LAR','fit_fano_MIN','fit_fano_MAX','fit_fano_BSL','SI']]
params = params.dropna()

indx = params[params['fit_fano_MIN'] == 0].index
params.drop(indx,inplace=True)
params['utype'] = ['multi'] * len(params.index)

#FFsuppression = -100 *((params['fit_fano_BSL'] - params['fit_fano_MIN']) / params['fit_fano_BSL'])
FFsuppression = -100 *(1-(params['fit_fano_MIN'] / params['fit_fano_BSL']))
FFsurfac = 100 * (params['fit_fano_LAR'] - params['fit_fano_RF'])/ params['fit_fano_RF']
params.insert(3,'FFsuppression',FFsuppression.values)
params.insert(3,'FFsurfac',FFsurfac.values)

plt.figure()
inds = params[params['FFsuppression'] < -300].index
params.drop(inds,inplace=True)
ax = plt.subplot(121)
SEM = params.groupby('layer')['FFsuppression'].sem()
params.groupby('layer')['FFsuppression'].mean().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-80,0)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsuppression',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-80,0)
if save_figures:
    plt.savefig(fig_dir + 'F2C-left.svg',bbox_inches='tight',pad_inches=0)

print('\n t-test FFsuppression LSG vs L4C')
print(sts.ttest_ind(params[params['layer'] == 'L4C']['FFsuppression'],params[params['layer'] == 'LSG']['FFsuppression']))
print('\n t-test FFsuppression LSG vs LIG')
print(sts.ttest_ind(params[params['layer'] == 'LIG']['FFsuppression'],params[params['layer'] == 'LSG']['FFsuppression']))

print('\n ANOVA: the effect of layer on FFsuppression')
lm = ols('FFsuppression ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print(params['FFsurfac'].min())
inds = params[params['FFsurfac'] > 300].index
params.drop(inds,inplace=True)

plt.figure()

ax = plt.subplot(121)
SEM = params.groupby('layer')['FFsurfac'].sem()
params.groupby('layer')['FFsurfac'].mean().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-70,160)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-70,160)

if save_figures:
    plt.savefig(fig_dir + 'F2C-right.svg',bbox_inches='tight',pad_inches=0)

# point stats and tests for each layer
print(params.groupby('layer')['FFsuppression'].mean())
print(params.groupby('layer')['FFsuppression'].sem())

print(params.groupby('layer')['FFsurfac'].mean())
print(params.groupby('layer')['FFsurfac'].sem())

print('\n ANOVA: the effect of layer on FFfacilitation')
lm = ols('FFsurfac ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print('\n t-test FFsurfac different from zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsurfac'],0)))

plt.figure()
ax = plt.subplot(111)
sns.scatterplot(x='SI',y='FFsurfac',hue='layer',data=params,ax=ax)
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xlabel('Suppression Index')
ax.set_ylabel('Fano Factor change (%)')
if save_figures:
    plt.savefig(fig_dir + 'F2D.svg',bbox_inches='tight',pad_inches=0)

print('\n test on correlation between FFsurfac and SI')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI'],df['FFsurfac'])))