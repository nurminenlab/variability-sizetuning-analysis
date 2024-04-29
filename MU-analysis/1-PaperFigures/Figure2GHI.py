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
params_df = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')

params = params_df[['layer',
                    'fit_fano_SML',
                    'fit_fano_RF',
                    'fit_fano_SUR',
                    'fit_fano_near_SUR_200',
                    'fit_fano_LAR',
                    'fit_fano_MIN',
                    'fit_fano_MAX',
                    'fit_fano_BSL',
                    'SI',
                    'SI_SUR',
                    'SI_SUR_2RF',
                    'animal']]
params = params.dropna()

params['utype'] = ['multi'] * len(params.index)

FFsuppression = -100 *(1-(params['fit_fano_RF'] / params['fit_fano_BSL']))
FFsurfac = 100 * (params['fit_fano_LAR'] - params['fit_fano_RF'])/ params['fit_fano_RF']
FFsurfac_SUR_2RF = 100 * (params['fit_fano_near_SUR_200'] - params['fit_fano_RF'])/ params['fit_fano_RF']
FFsurfac_SUR = 100 * (params['fit_fano_SUR'] - params['fit_fano_RF'])/ params['fit_fano_RF']
params.insert(3,'FFsuppression',FFsuppression.values)
params.insert(3,'FFsurfac',FFsurfac.values)
params.insert(3,'FFsurfac_SUR',FFsurfac_SUR.values)
params.insert(3,'FFsurfac_SUR_2RF',FFsurfac_SUR_2RF.values)

G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')
SG = params.query('layer == "LSG"')

################
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


################

plt.figure() # per neuron surround
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
    plt.savefig(fig_dir + 'F2G-SUR.svg',bbox_inches='tight',pad_inches=0)

print('\n test on correlation between FFsurfac_SUR and SI_SUR')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI_SUR'],df['FFsurfac_SUR'])))

################

plt.figure() # per neuron near-surround

Y = SG['FFsurfac_SUR_2RF'].values
X = SG['SI_SUR_2RF'].values
X = sm.add_constant(X)
SG_results = sm.OLS(Y,X).fit()

Y = G['FFsurfac_SUR_2RF'].values
X = G['SI_SUR_2RF'].values
X = sm.add_constant(X)
G_results = sm.OLS(Y,X).fit()

Y = IG['FFsurfac_SUR_2RF'].values
X = IG['SI_SUR_2RF'].values
X = sm.add_constant(X)
IG_results = sm.OLS(Y,X).fit()

ax = plt.subplot(111)
sns.scatterplot(x='SI_SUR_2RF',y='FFsurfac_SUR_2RF',hue='layer',style='animal',data=params,ax=ax)
ax.plot([np.min(SG['SI_SUR_2RF']),np.max(SG['SI_SUR_2RF'])],
        SG_results.params[0] + SG_results.params[1]*np.array([np.min(SG['SI_SUR_2RF']),np.max(SG['SI_SUR_2RF'])]),'b-')
ax.plot([np.min(G['SI_SUR_2RF']),np.max(G['SI_SUR_2RF'])],
        G_results.params[0] + G_results.params[1]*np.array([np.min(G['SI_SUR_2RF']),np.max(G['SI_SUR_2RF'])]),'r-')
ax.plot([np.min(IG['SI_SUR_2RF']),np.max(IG['SI_SUR_2RF'])],
        IG_results.params[0] + IG_results.params[1]*np.array([np.min(IG['SI_SUR_2RF']),np.max(IG['SI_SUR_2RF'])]),'g-')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xlabel('Suppression Index')
ax.set_ylabel('Fano Factor change (%)')
if save_figures:
    plt.savefig(fig_dir + 'F2G-SUR-2RF.svg',bbox_inches='tight',pad_inches=0)

print('\n test on correlation between FFsurfac_SUR_2RF and SI_SUR_2RF')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI_SUR_2RF'],df['FFsurfac_SUR_2RF'])))



