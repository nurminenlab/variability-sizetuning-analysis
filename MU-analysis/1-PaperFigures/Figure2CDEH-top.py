import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params_df = pd.read_csv(F_dir + 'extracted_params-newselection-Jun2023.csv')

params = params_df[['layer','fit_fano_SML','fit_fano_RF','fit_fano_SUR','fit_fano_LAR','fit_fano_MIN','fit_fano_MAX','fit_fano_BSL','SI','SI_SUR','animal']]
params = params.dropna()

params['utype'] = ['multi'] * len(params.index)

FFsuppression = -100 *(1-(params['fit_fano_RF'] / params['fit_fano_BSL']))
FFsurfac = 100 * (params['fit_fano_LAR'] - params['fit_fano_RF'])/ params['fit_fano_RF']
FFsurfac_SUR = 100 * (params['fit_fano_SUR'] - params['fit_fano_RF'])/ params['fit_fano_RF']
params.insert(3,'FFsuppression',FFsuppression.values)
params.insert(3,'FFsurfac',FFsurfac.values)
params.insert(3,'FFsurfac_SUR',FFsurfac_SUR.values)

# fano suppression RF
plt.figure()
inds = params[params['FFsuppression'] < -300].index
params.drop(inds,inplace=True)

G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')
SG = params.query('layer == "LSG"')

ciG  = sts.bootstrap((G['FFsuppression'].values,),np.median,confidence_level=0.68)
ciIG = sts.bootstrap((IG['FFsuppression'].values,),np.median,confidence_level=0.68)
ciSG = sts.bootstrap((SG['FFsuppression'].values,),np.median,confidence_level=0.68)

SEM = [ciG.standard_error, ciIG.standard_error, ciSG.standard_error]

ax = plt.subplot(121)
params.groupby('layer')['FFsuppression'].median().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-80,90)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsuppression',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-80,90)
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

ciG  = sts.bootstrap((G['FFsurfac'].values,),np.median,confidence_level=0.68)
ciIG = sts.bootstrap((IG['FFsurfac'].values,),np.median,confidence_level=0.68)
ciSG = sts.bootstrap((SG['FFsurfac'].values,),np.median,confidence_level=0.68)

SEM = [ciG.standard_error, ciIG.standard_error, ciSG.standard_error]

params.groupby('layer')['FFsurfac'].median().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-70,160)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac',hue='animal',data=params,ax=ax,size=3)
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

SG = params.query('layer == "LSG"')
G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')

Y = SG['FFsurfac'].values
X = SG['SI'].values
X = sm.add_constant(X)
SG_results = sm.OLS(Y,X).fit()

Y = G['FFsurfac'].values
X = G['SI'].values
X = sm.add_constant(X)
G_results = sm.OLS(Y,X).fit()

Y = IG['FFsurfac'].values
X = IG['SI'].values
X = sm.add_constant(X)
IG_results = sm.OLS(Y,X).fit()

plt.figure()
ax = plt.subplot(111)
sns.scatterplot(x='SI',y='FFsurfac',hue='layer',style='animal',data=params,ax=ax)
ax.plot([np.min(SG['SI']),np.max(SG['SI'])],
        SG_results.params[0] + SG_results.params[1]*np.array([np.min(SG['SI']),np.max(SG['SI'])]),'b-')
ax.plot([np.min(G['SI']),np.max(G['SI'])],
        G_results.params[0] + G_results.params[1]*np.array([np.min(G['SI']),np.max(G['SI'])]),'r-')
ax.plot([np.min(IG['SI']),np.max(IG['SI'])],
        IG_results.params[0] + IG_results.params[1]*np.array([np.min(IG['SI']),np.max(IG['SI'])]),'g-')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xlabel('Suppression Index')
ax.set_ylabel('Fano Factor change (%)')
if save_figures:
    plt.savefig(fig_dir + 'F2G.svg',bbox_inches='tight',pad_inches=0)

print('\n test on correlation between FFsurfac and SI')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI'],df['FFsurfac'])))


plt.figure()
ax = plt.subplot(121)
 
G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')
SG = params.query('layer == "LSG"')

ciG  = sts.bootstrap((G['FFsurfac_SUR'].values,),np.median,confidence_level=0.68)
ciIG = sts.bootstrap((IG['FFsurfac_SUR'].values,),np.median,confidence_level=0.68)
ciSG = sts.bootstrap((SG['FFsurfac_SUR'].values,),np.median,confidence_level=0.68)

SEM = [ciG.standard_error, ciIG.standard_error, ciSG.standard_error]

params.groupby('layer')['FFsurfac_SUR'].median().plot(kind='bar',ax=ax,yerr=SEM,color='white',edgecolor='red')
ax.set_ylim(-70,160)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac_SUR',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-70,160)

if save_figures:
    plt.savefig(fig_dir + 'F2E-top.svg',bbox_inches='tight',pad_inches=0)

print(params.groupby('layer')['FFsurfac_SUR'].mean())
print(params.groupby('layer')['FFsurfac_SUR'].sem())

print('\n ANOVA: the effect of layer on FFfacilitation')
lm = ols('FFsurfac_SUR ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print('\n t-test FFsurfac_SUR different from zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsurfac_SUR'],0)))

SG = params.query('layer == "LSG"')
G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')

Y = SG['FFsurfac_SUR'].values
X = SG['SI_SUR'].values
X = sm.add_constant(X)
SG_results = sm.OLS(Y,X).fit()

Y = G['FFsurfac_SUR'].values
X = G['SI_SUR'].values
X = sm.add_constant(X)
G_results = sm.OLS(Y,X).fit()

Y = IG['FFsurfac_SUR'].values
X = IG['SI_SUR'].values
X = sm.add_constant(X)
IG_results = sm.OLS(Y,X).fit()

plt.figure()
ax = plt.subplot(111)
sns.scatterplot(x='SI_SUR',y='FFsurfac_SUR',hue='layer',style='animal',data=params,ax=ax)
ax.plot([np.min(SG['SI_SUR']),np.max(SG['SI_SUR'])],
        SG_results.params[0] + SG_results.params[1]*np.array([np.min(SG['SI_SUR']),np.max(SG['SI_SUR'])]),'b-')
ax.plot([np.min(G['SI_SUR']),np.max(G['SI_SUR'])],
        G_results.params[0] + G_results.params[1]*np.array([np.min(G['SI_SUR']),np.max(G['SI_SUR'])]),'r-')
ax.plot([np.min(IG['SI_SUR']),np.max(IG['SI_SUR'])],
        IG_results.params[0] + IG_results.params[1]*np.array([np.min(IG['SI_SUR']),np.max(IG['SI_SUR'])]),'g-')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xlabel('Suppression Index')
ax.set_ylabel('Fano Factor change (%)')
if save_figures:
    plt.savefig(fig_dir + 'F2G.svg',bbox_inches='tight',pad_inches=0)

print('\n test on correlation between FFsurfac_SUR and SI_SUR')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI_SUR'],df['FFsurfac_SUR'])))