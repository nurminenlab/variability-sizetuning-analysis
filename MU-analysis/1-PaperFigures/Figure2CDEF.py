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

def bootstrap_median_test(sample, num_bootstrap=10000, alpha=0.05):
    """
    Perform a bootstrap test to check if the median of the sample differs from zero.

    Parameters:
    sample (array-like): The data sample.
    num_bootstrap (int): The number of bootstrap samples to generate.
    alpha (float): The significance level for the confidence interval.

    Returns:
    bool: True if the median differs from zero, False otherwise.
    float: The lower bound of the confidence interval.
    float: The upper bound of the confidence interval.
    float: The p-value of the test.
    """
    n = len(sample)
    medians = np.zeros(num_bootstrap)

    # Generate bootstrap samples and compute medians
    for i in range(num_bootstrap):
        bootstrap_sample = np.random.choice(sample, size=n, replace=True)
        medians[i] = np.median(bootstrap_sample)

    # Compute the confidence interval
    lower_bound = np.percentile(medians, 100 * alpha / 2)
    upper_bound = np.percentile(medians, 100 * (1 - alpha / 2))

    # Check if zero is within the confidence interval
    differs_from_zero = not (lower_bound <= 0 <= upper_bound)

    # Compute the p-value
    p_value = np.mean(medians <= 0) if np.median(sample) > 0 else np.mean(medians >= 0)

    return differs_from_zero, p_value


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

# fano suppression RF
plt.figure()
inds = params[params['FFsuppression'] < -300].index
params.drop(inds,inplace=True)

G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')
SG = params.query('layer == "LSG"')

SEM_ffsupr = params.groupby('layer')['FFsuppression'].sem()
SEM_ffsupr['L4C'] = sts.bootstrap((G['FFsuppression'].values,),np.median,confidence_level=0.68).standard_error
SEM_ffsupr['LIG'] = sts.bootstrap((IG['FFsuppression'].values,),np.median,confidence_level=0.68).standard_error
SEM_ffsupr['LSG'] = sts.bootstrap((SG['FFsuppression'].values,),np.median,confidence_level=0.68).standard_error

ax = plt.subplot(121)
params.groupby('layer')['FFsuppression'].median().plot(kind='bar',ax=ax,yerr=SEM_ffsupr,color='white',edgecolor='red')
ax.set_ylim(-80,90)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsuppression',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-80,90)
if save_figures:
    plt.savefig(fig_dir + 'F2C-left.svg',bbox_inches='tight',pad_inches=0)


# computer significance of fano suppression in RF
SG_differs_from_zero, SG_p_value = bootstrap_median_test(SG['FFsuppression'].values)
G_differs_from_zero, G_p_value = bootstrap_median_test(G['FFsuppression'].values)
IG_differs_from_zero, IG_p_value = bootstrap_median_test(IG['FFsuppression'].values)

print('\n t-test FFsuppression LSG vs L4C')
print(sts.ttest_ind(params[params['layer'] == 'L4C']['FFsuppression'],params[params['layer'] == 'LSG']['FFsuppression']))
print('\n t-test FFsuppression LSG vs LIG')
print(sts.ttest_ind(params[params['layer'] == 'LIG']['FFsuppression'],params[params['layer'] == 'LSG']['FFsuppression']))

#print('\n ANOVA: the effect of layer on FFsuppression')
#lm = ols('FFsuppression ~ C(layer)',data=params).fit()
#print(sm.stats.anova_lm(lm,typ=1))

print('FFsuppression medians',params.groupby('layer')['FFsuppression'].median())
print('FFsuppression SEM',SEM_ffsupr)

print(params['FFsurfac'].min())
inds = params[params['FFsurfac'] > 300].index
params.drop(inds,inplace=True)

# set the same median for each layer
SG['FFsuppression_zero_median'] = SG['FFsuppression'] - SG['FFsuppression'].median()
G['FFsuppression_zero_median']  = G['FFsuppression'] - G['FFsuppression'].median()
IG['FFsuppression_zero_median'] = IG['FFsuppression'] - IG['FFsuppression'].median()

SG_distr = sts.bootstrap((SG['FFsuppression_zero_median'].values,),np.median).bootstrap_distribution
G_distr  = sts.bootstrap((G['FFsuppression_zero_median'].values,),np.median).bootstrap_distribution
IG_distr = sts.bootstrap((IG['FFsuppression_zero_median'].values,),np.median).bootstrap_distribution

SG_G  = np.abs(SG_distr - G_distr)
SG_IG = np.abs(SG_distr - IG_distr)
G_IG  = np.abs(G_distr - IG_distr)

print('FFsuppression SG vs L4C: p-value')
print(np.sum(SG_G > np.abs(SG['FFsuppression'].median() - G['FFsuppression'].median())) / len(SG_G))

print('FFsuppression SG vs IG: p-value')
print(np.sum(SG_IG > np.abs(SG['FFsuppression'].median() - IG['FFsuppression'].median())) / len(SG_IG))

print('FFsuppression IG vs L4C: p-value')  
print(np.sum(SG_IG > np.abs(IG['FFsuppression'].median() - G['FFsuppression'].median())) / len(G_IG))

#################
# FF surfac
plt.figure()
ax = plt.subplot(121)

# container for SEM
SEM_ffsurfac = params.groupby('layer')['FFsurfac'].sem()
SEM_ffsurfac['L4C'] = sts.bootstrap((G['FFsurfac'].values,),np.median,confidence_level=0.68).standard_error
SEM_ffsurfac['LIG'] = sts.bootstrap((IG['FFsurfac'].values,),np.median,confidence_level=0.68).standard_error
SEM_ffsurfac['LSG'] = sts.bootstrap((SG['FFsurfac'].values,),np.median,confidence_level=0.68).standard_error

params.groupby('layer')['FFsurfac'].median().plot(kind='bar',ax=ax,yerr=SEM_ffsurfac,color='white',edgecolor='red')
ax.set_ylim(-70,160)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac',hue='animal',data=params,ax=ax,size=3)
ax.set_ylim(-70,160)

if save_figures:
    plt.savefig(fig_dir + 'F2C-right.svg',bbox_inches='tight',pad_inches=0)


#print('\n t-test FFsurfac different from zero')
#print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsurfac'],0)))

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

print('\n test on correlation between FFsurfac and SI')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI'],df['FFsurfac'])))

print('FFsurfac medians',params.groupby('layer')['FFsurfac'].median())
print('FFsurfac SEM',SEM_ffsurfac)

# set the same median for each layer
SG['FFsurfac_zero_median'] = SG['FFsurfac'] - SG['FFsurfac'].median()
G['FFsurfac_zero_median']  = G['FFsurfac'] - G['FFsurfac'].median()
IG['FFsurfac_zero_median'] = IG['FFsurfac'] - IG['FFsurfac'].median()

SG_distr = sts.bootstrap((SG['FFsurfac_zero_median'].values,),np.median).bootstrap_distribution
G_distr  = sts.bootstrap((G['FFsurfac_zero_median'].values,),np.median).bootstrap_distribution
IG_distr = sts.bootstrap((IG['FFsurfac_zero_median'].values,),np.median).bootstrap_distribution

print('FFsurfac SG larger than zero: p-value')
print(np.sum(SG_distr > SG['FFsurfac'].median()) / len(SG_distr))

print('FFsurfac G larger than zero: p-value')
print(np.sum(G_distr > G['FFsurfac'].median()) / len(G_distr))

print('FFsurfac IG larger than zero: p-value')
print(np.sum(IG_distr < IG['FFsurfac'].median()) / len(IG_distr))

plt.figure()
ax = plt.subplot(121)
 
G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')
SG = params.query('layer == "LSG"')

# container for SEM
SEM_ffsurfac_SUR = params.groupby('layer')['FFsurfac_SUR'].sem()
SEM_ffsurfac_SUR['L4C'] = sts.bootstrap((G['FFsurfac_SUR'].values,),np.median).standard_error
SEM_ffsurfac_SUR['LIG'] = sts.bootstrap((IG['FFsurfac_SUR'].values,),np.median,confidence_level=0.68).standard_error
SEM_ffsurfac_SUR['LSG'] = sts.bootstrap((SG['FFsurfac_SUR'].values,),np.median,confidence_level=0.68).standard_error

params.groupby('layer')['FFsurfac_SUR'].median().plot(kind='bar',ax=ax,yerr=SEM_ffsurfac_SUR,color='white',edgecolor='red')
ax.set_ylim(-70,160)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac_SUR',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-70,160)

if save_figures:
    plt.savefig(fig_dir + 'F2E-top-SUR.svg',bbox_inches='tight',pad_inches=0)


print('\n ANOVA: the effect of layer on FFfacilitation')
lm = ols('FFsurfac_SUR ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print('\n t-test FFsurfac_SUR different from zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsurfac_SUR'],0)))

SG = params.query('layer == "LSG"')
G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')


print('FFsurfac_SUR medians',params.groupby('layer')['FFsurfac_SUR'].median())
print('FFsurfac_SUR SEM',SEM_ffsurfac_SUR)

# set the same median for each layer
SG['FFsurfac_SUR_zero_median'] = SG['FFsurfac_SUR'] - SG['FFsurfac_SUR'].median()
G['FFsurfac_SUR_zero_median']  = G['FFsurfac_SUR'] - G['FFsurfac_SUR'].median()
IG['FFsurfac_SUR_zero_median'] = IG['FFsurfac_SUR'] - IG['FFsurfac_SUR'].median()

SG_distr = sts.bootstrap((SG['FFsurfac_SUR_zero_median'].values,),np.median).bootstrap_distribution
G_distr  = sts.bootstrap((G['FFsurfac_SUR_zero_median'].values,),np.median).bootstrap_distribution
IG_distr = sts.bootstrap((IG['FFsurfac_SUR_zero_median'].values,),np.median).bootstrap_distribution

print('FFsurfac_SUR SG > 0 : p-value')
print(np.sum(SG_distr > np.abs(SG['FFsurfac_SUR'].median())) / len(SG_distr))

print('FFsurfac_SUR G > 0 : p-value')
print(np.sum(G_distr > np.abs(G['FFsurfac_SUR'].median())) / len(G_distr))

print('FFsurfac_SUR IG > 0 : p-value')
print(np.sum(IG_distr > np.abs(IG['FFsurfac_SUR'].median())) / len(IG_distr))

plt.figure()
ax = plt.subplot(121)
 
G  = params.query('layer == "L4C"')
IG = params.query('layer == "LIG"')
SG = params.query('layer == "LSG"')

# container for SEM
SEM_ffsurfac_SUR = params.groupby('layer')['FFsurfac_SUR_2RF'].sem()
SEM_ffsurfac_SUR['L4C'] = sts.bootstrap((G['FFsurfac_SUR_2RF'].values,),np.median).standard_error
SEM_ffsurfac_SUR['LIG'] = sts.bootstrap((IG['FFsurfac_SUR_2RF'].values,),np.median,confidence_level=0.68).standard_error
SEM_ffsurfac_SUR['LSG'] = sts.bootstrap((SG['FFsurfac_SUR_2RF'].values,),np.median,confidence_level=0.68).standard_error

params.groupby('layer')['FFsurfac_SUR_2RF'].median().plot(kind='bar',ax=ax,yerr=SEM_ffsurfac_SUR,color='white',edgecolor='red')
ax.set_ylim(-70,160)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac_SUR_2RF',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-70,160)

if save_figures:
    plt.savefig(fig_dir + 'F2E-top-SUR-2RF.svg',bbox_inches='tight',pad_inches=0)


print('\n ANOVA: the effect of layer on FFfacilitation')
lm = ols('FFsurfac_SUR ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print('\n t-test FFsurfac_SUR_2RF different from zero')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['FFsurfac_SUR_2RF'],0)))

print('\n test on correlation between FFsurfac_SUR_2RF and SI_SUR_2RF')
print(params.groupby('layer').apply(lambda df: sts.pearsonr(df['SI_SUR_2RF'],df['FFsurfac_SUR_2RF'])))

print('FFsurfac_SUR_2RF medians',params.groupby('layer')['FFsurfac_SUR_2RF'].median())
print('FFsurfac_SUR_2RF SEM',SEM_ffsurfac_SUR)

# set the same median for each layer
SG['FFsurfac_SUR_2RF_zero_median'] = SG['FFsurfac_SUR_2RF'] - SG['FFsurfac_SUR_2RF'].median()
G['FFsurfac_SUR_2RF_zero_median']  = G['FFsurfac_SUR_2RF'] - G['FFsurfac_SUR_2RF'].median()
IG['FFsurfac_SUR_2RF_zero_median'] = IG['FFsurfac_SUR_2RF'] - IG['FFsurfac_SUR_2RF'].median()

SG_distr = sts.bootstrap((SG['FFsurfac_SUR_2RF_zero_median'].values,),np.median).bootstrap_distribution
G_distr  = sts.bootstrap((G['FFsurfac_SUR_2RF_zero_median'].values,),np.median).bootstrap_distribution
IG_distr = sts.bootstrap((IG['FFsurfac_SUR_2RF_zero_median'].values,),np.median).bootstrap_distribution

print('FFsurfac_SUR_2RF SG > 0 : p-value')
print(np.sum(SG_distr > np.abs(SG['FFsurfac_SUR_2RF'].median())) / len(SG_distr))

print('FFsurfac_SUR_2RF G > 0 : p-value')
print(np.sum(G_distr > np.abs(G['FFsurfac_SUR_2RF'].median())) / len(G_distr))

print('FFsurfac_SUR_2RF IG > 0 : p-value')
print(np.sum(IG_distr > np.abs(IG['FFsurfac_SUR_2RF'].median())) / len(IG_distr))

