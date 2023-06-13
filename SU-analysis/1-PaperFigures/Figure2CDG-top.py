import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
fig_dir = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'

params = pd.read_csv(F_dir + 'SU-extracted_params-Jun2023.csv')

#params = params.dropna()

#indx = params[params['fit_fano_MIN'] == 0].index
#params.drop(indx,inplace=True)
params['utype'] = ['multi'] * len(params.index)

FFsuppression = -100*((params['fit_fano_BSL']-params['fit_fano_RF'])) / params['fit_fano_BSL']
FFsurfac =  100*(params['fit_fano_LAR'] - params['fit_fano_RF']) / params['fit_fano_RF']
params.insert(3,'FFchange_BSL_RF',FFsuppression.values)
params.insert(3,'FFchange_RF_large',FFsurfac.values)

plt.figure()
inds = params[params['FFchange_BSL_RF'] < -300].index
params.drop(inds,inplace=True)
ax = plt.subplot(121)
SEM = params.groupby('layer')['FFchange_BSL_RF'].sem()
params.groupby('layer')['FFchange_BSL_RF'].mean().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-80,250)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFchange_BSL_RF',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-80,250)
if save_figures:
    plt.savefig(fig_dir + 'F2C-left.svg',bbox_inches='tight',pad_inches=0)

print('\n t-test FFsuppression LSG vs L4C')
print(sts.ttest_ind(params[params['layer'] == 'L4C']['FFchange_BSL_RF'],params[params['layer'] == 'LSG']['FFchange_BSL_RF']))
print('\n t-test FFsuppression LSG vs LIG')
print(sts.ttest_ind(params[params['layer'] == 'LIG']['FFchange_BSL_RF'],params[params['layer'] == 'LSG']['FFchange_BSL_RF']))

print('\n ANOVA: the effect of layer on FFsuppression')
lm = ols('FFsuppression ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print(params['FFchange_RF_large'].min())
inds = params[params['FFchange_RF_large'] > 300].index
params.drop(inds,inplace=True)

plt.figure()

ax = plt.subplot(121)
SEM = params.groupby('layer')['FFchange_RF_large'].sem()
params.groupby('layer')['FFchange_RF_large'].mean().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-70,160)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFchange_RF_large',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-70,160)

if save_figures:
    plt.savefig(fig_dir + 'F2C-right.svg',bbox_inches='tight',pad_inches=0)

# point stats and tests for each layer
print(params.groupby('layer')['FFchange_BSL_RF'].mean())
print(params.groupby('layer')['FFchange_BSL_RF'].sem())

print(params.groupby('layer')['FFchange_RF_large'].mean())
print(params.groupby('layer')['FFchange_RF_large'].sem())

print('\n ANOVA: the effect of layer on FFfacilitation')
lm = ols('FFchange_RF_large ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print('\n t-test FFsurfac different from zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFchange_RF_large'],0)))

SG = params.query('layer == "LSG"')
G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')

Y = SG['FFchange_RF_large'].values
X = SG['SI'].values
X = sm.add_constant(X)
SG_results = sm.OLS(Y,X).fit()

Y = G['FFchange_RF_large'].values
X = G['SI'].values
X = sm.add_constant(X)
G_results = sm.OLS(Y,X).fit()

Y = IG['FFchange_RF_large'].values
X = IG['SI'].values
X = sm.add_constant(X)
IG_results = sm.OLS(Y,X).fit()

plt.figure()
ax = plt.subplot(111)
sns.scatterplot(x='SI',y='FFchange_RF_large',hue='layer',data=params,ax=ax)
""" ax.plot([np.min(SG['SI']),np.max(SG['SI'])],
        SG_results.params[0] + SG_results.params[1]*np.array([np.min(SG['SI']),np.max(SG['SI'])]),'b-')
ax.plot([np.min(G['SI']),np.max(G['SI'])],
        G_results.params[0] + G_results.params[1]*np.array([np.min(G['SI']),np.max(G['SI'])]),'r-')
ax.plot([np.min(IG['SI']),np.max(IG['SI'])],
        IG_results.params[0] + IG_results.params[1]*np.array([np.min(IG['SI']),np.max(IG['SI'])]),'g-') """
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xlabel('Suppression Index')
ax.set_ylabel('Fano Factor change (%)')
if save_figures:
    plt.savefig(fig_dir + 'F2G.svg',bbox_inches='tight',pad_inches=0)

print('\n test on correlation between FFsurfac and SI')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI'],df['FFchange_RF_large'])))

