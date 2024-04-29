import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params = pd.read_csv(F_dir + 'FA-params-Aug-2023.csv')

params = params[['layer','animal','fit_FA_SML','fit_FA_RF','fit_FA_SUR','fit_FA_LAR','fit_FA_MIN','fit_FA_MAX','fit_FA_BSL']]
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
# bootstrap errors 
G  = params.query('layer == "G"')
IG = params.query('layer == "IG"')
SG = params.query('layer == "SG"')

SEM_ffsupr = params.groupby('layer')['FFsuppression'].sem() # create appropriate table to store bootstrapped errors
SEM_ffsupr['G'] = sts.bootstrap((G['FFsuppression'].values,),np.median).standard_error
SEM_ffsupr['IG'] = sts.bootstrap((IG['FFsuppression'].values,),np.median).standard_error
SEM_ffsupr['SG'] = sts.bootstrap((SG['FFsuppression'].values,),np.median).standard_error

inds = params[params['FFsuppression'] > 300].index
params.drop(inds,inplace=True)
ax = plt.subplot(121)
params.groupby('layer')['FFsuppression'].median().plot(kind='bar',ax=ax,yerr=SEM_ffsupr,color='white',edgecolor='red')

ax.set_ylim(-100,200)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsuppression',hue='animal',data=params,ax=ax,size=3,color='red')
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

SEM_ffsurfac = params.groupby('layer')['FFsurfac'].sem() # create appropriate table to store bootstrapped errors
SEM_ffsurfac['G'] = sts.bootstrap((G['FFsurfac'].values,),np.median).standard_error
SEM_ffsurfac['IG'] = sts.bootstrap((IG['FFsurfac'].values,),np.median).standard_error
SEM_ffsurfac['SG'] = sts.bootstrap((SG['FFsurfac'].values,),np.median).standard_error

ax = plt.subplot(121)
params.groupby('layer')['FFsurfac'].median().plot(kind='bar',ax=ax,yerr=SEM_ffsurfac,color='white',edgecolor='red')
ax.set_ylim(-100,300)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-100,300)

if save_figures:
    plt.savefig(fig_dir + 'F5C-right.svg',bbox_inches='tight',pad_inches=0)

# point stats and tests for each layer
print(params.groupby('layer')['FFsuppression'].median())
print(SEM_ffsupr)

# create zero median distributions for bootstrap
SG_zeromedian = SG['FFsuppression'] - SG['FFsuppression'].median()
G_zeromedian  = G['FFsuppression'] - G['FFsuppression'].median()
IG_zeromedian = IG['FFsuppression'] - IG['FFsuppression'].median()

SG.insert(3,'FFsuppression_zeromedian',SG_zeromedian.values)
G.insert(3,'FFsuppression_zeromedian',G_zeromedian.values)
IG.insert(3,'FFsuppression_zeromedian',IG_zeromedian.values)

SG_boot = sts.bootstrap((SG['FFsuppression_zeromedian'].values,),np.median).bootstrap_distribution
G_boot  = sts.bootstrap((G['FFsuppression_zeromedian'].values,),np.median).bootstrap_distribution
IG_boot = sts.bootstrap((IG['FFsuppression_zeromedian'].values,),np.median).bootstrap_distribution

print('SG p: ', np.sum(SG_boot < SG['FFsuppression'].median())/len(SG_boot))
print('G p: ', np.sum(G_boot < G['FFsuppression'].median())/len(G_boot))
print('IG p: ', np.sum(IG_boot < IG['FFsuppression'].median())/len(IG_boot))


# point stats and tests for surround effects
print(params.groupby('layer')['FFsurfac'].median())
print(SEM_ffsurfac)

# create zero median distributions for bootstrap
SG_zeromedian = SG['FFsurfac'] - SG['FFsurfac'].median()
G_zeromedian  = G['FFsurfac'] - G['FFsurfac'].median()
IG_zeromedian = IG['FFsurfac'] - IG['FFsurfac'].median()

SG.insert(3,'FFsurfac_zeromedian',SG_zeromedian.values)
G.insert(3,'FFsurfac_zeromedian',G_zeromedian.values)
IG.insert(3,'FFsurfac_zeromedian',IG_zeromedian.values)

SG_boot = sts.bootstrap((SG['FFsurfac_zeromedian'].values,),np.median).bootstrap_distribution
G_boot  = sts.bootstrap((G['FFsurfac_zeromedian'].values,),np.median).bootstrap_distribution
IG_boot = sts.bootstrap((IG['FFsurfac_zeromedian'].values,),np.median).bootstrap_distribution

print('SG p: ', np.sum(SG_boot > SG['FFsurfac'].median())/len(SG_boot))
print('G p: ', np.sum(G_boot > G['FFsurfac'].median())/len(G_boot))
print('IG p: ', np.sum(IG_boot < IG['FFsurfac'].median())/len(IG_boot))
