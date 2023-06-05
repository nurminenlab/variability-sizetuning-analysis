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
params = pd.read_csv(F_dir + 'FA-params-Oct-2022.csv')

params = params[['layer','fit_FA_SML','fit_FA_RF','fit_FA_SUR','fit_FA_LAR','fit_FA_MIN','fit_FA_MAX','fit_FA_BSL']]
params = params.dropna()

FF_size = pd.DataFrame(columns=['FA','size','layer'])
FF_RF = pd.DataFrame(data={'FA':params['fit_FA_RF'].values,
                        'size':['RF']*len(params.index),
                        'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'FA':params['fit_FA_LAR'].values,
                        'size':['LAR']*len(params.index),
                        'layer':params['layer'].values})

FF_size = FF_size.append(FF_RF)
FF_size = FF_size.append(FF_LAR)

plt.figure()
ax = plt.subplot(111)

SEM = params.groupby('layer')[['fit_FA_SML','fit_FA_RF','fit_FA_SUR','fit_FA_LAR']].sem()
params.groupby('layer')[['fit_FA_SML','fit_FA_RF','fit_FA_SUR','fit_FA_LAR']].mean().plot(ax=ax,kind='bar',yerr=SEM)
if save_figures:
    plt.savefig(fig_dir + 'F5B-top.svg',bbox_inches='tight',pad_inches=0)


plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='FA',hue='size',data=FF_size,ax=ax,size=3,dodge=True)
if save_figures:
    plt.savefig(fig_dir + 'F5B-bottom.svg',bbox_inches='tight',pad_inches=0)


FFsuppression = -100 *(1-(params['fit_FA_RF'] / params['fit_FA_BSL']))
FFsurfac = 100 * (params['fit_FA_LAR'] - params['fit_FA_RF'])/ params['fit_FA_RF']
params.insert(3,'FFsuppression',FFsuppression.values)
params.insert(3,'FFsurfac',FFsurfac.values)

plt.figure()
inds = params[params['FFsuppression'] > 300].index
params.drop(inds,inplace=True)
ax = plt.subplot(121)
SEM = params.groupby('layer')['FFsuppression'].sem()

params.groupby('layer')['FFsuppression'].mean().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-100,200)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsuppression',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-100,200)
if save_figures:
    plt.savefig(fig_dir + 'F5C-left.svg',bbox_inches='tight',pad_inches=0)

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
#ax.set_ylim(-100,300)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac',data=params,ax=ax,size=3,color='red')
#ax.set_ylim(-100,300)

if save_figures:
    plt.savefig(fig_dir + 'F5C-right.svg',bbox_inches='tight',pad_inches=0)

# point stats and tests for each layer
print(params.groupby('layer')['FFsuppression'].mean())
print(params.groupby('layer')['FFsuppression'].sem())

print(params.groupby('layer')['FFsurfac'].mean())
print(params.groupby('layer')['FFsurfac'].sem())

print('\n ANOVA: the effect of layer on FFfacilitation')
lm = ols('FFsurfac ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print('\n t-test FFsurfac larger than zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsurfac'],0,alternative='greater')))

print('\n t-test FFsurfac smaller than zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsurfac'],0,alternative='less')))

print('\n t-test FFsupp smaller than zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsuppression'],0,alternative='less')))

