import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'

params = pd.read_csv(F_dir + 'SU-extracted_params-Jul2023.csv')

# we clean up units without much fano factor tuning
SG_units_to_remove = [7,14,26,50,51,53,58,68,80]
IG_units_to_remove = [20,46,81]

idx_to_remove = []
for unit in SG_units_to_remove:
    idx_to_remove.append(params[params['unit'] == unit].index[0])

for unit in IG_units_to_remove:
    idx_to_remove.append(params[params['unit'] == unit].index[0])

params.drop(idx_to_remove,axis=0,inplace=True)
params.drop(params[params['layer'] == 'L4C'].index,axis=0,inplace=True) # just one L4C unit

FF_size = pd.DataFrame(columns=['fano','size','layer'])
FF_size_all = pd.DataFrame(columns=['fano','size','layer'])

params['fit_fano_SML_nrmd'] = params['fit_fano_SML'] / params['fit_fano_RF']
params['fit_fano_near_SUR_200_nrmd'] = params['fit_fano_near_SUR_200'] / params['fit_fano_RF']
params['fit_fano_far_SUR_400_nrmd'] = params['fit_fano_far_SUR_400'] / params['fit_fano_RF']
params['fit_fano_far_SUR_800_nrmd'] = params['fit_fano_far_SUR_800'] / params['fit_fano_RF']

params['fit_fano_LAR_nrmd'] = params['fit_fano_LAR'] / params['fit_fano_RF']
params['fit_fano_RF_nrmd'] = params['fit_fano_RF'] / params['fit_fano_RF']

FF_RF = pd.DataFrame(data={'fano':params['fit_fano_RF_nrmd'].values,'size':['RF']*len(params.index),'layer':params['layer'].values})
FF_near_SUR_200 = pd.DataFrame(data={'fano':params['fit_fano_near_SUR_200_nrmd'].values,'size':['near_SUR_200']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_400 = pd.DataFrame(data={'fano':params['fit_fano_far_SUR_400_nrmd'].values,'size':['far_SUR_400']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_800 = pd.DataFrame(data={'fano':params['fit_fano_far_SUR_800_nrmd'].values,'size':['far_SUR_800']*len(params.index),'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'fano':params['fit_fano_LAR'].values,'size':['LAR']*len(params.index),'layer':params['layer'].values})

FF_size = FF_size.append(FF_RF)
FF_size = FF_size.append(FF_near_SUR_200)
FF_size = FF_size.append(FF_far_SUR_400)
FF_size = FF_size.append(FF_far_SUR_800)
FF_size = FF_size.append(FF_LAR)

SG_df = params.query('layer == "LSG"')
IG_df = params.query('layer == "LIG"')

plt.figure()
ax = plt.subplot(111)
# replace with bootstrap
SEM = params.groupby('layer')[['fit_fano_SML_nrmd',
                               'fit_fano_RF_nrmd',
                               'fit_fano_near_SUR_200_nrmd',                               
                               'fit_fano_far_SUR_400_nrmd',                               
                               'fit_fano_far_SUR_800_nrmd',
                               'fit_fano_LAR_nrmd']].sem()

params.groupby('layer')[['fit_fano_SML_nrmd',
                        'fit_fano_RF_nrmd',
                        'fit_fano_near_SUR_200_nrmd',                        
                        'fit_fano_far_SUR_400_nrmd',                        
                        'fit_fano_far_SUR_800_nrmd',
                        'fit_fano_LAR_nrmd']].median().plot(ax=ax,kind='bar',yerr=SEM)


if save_figures:
    plt.savefig(fig_dir + 'F2B-top.svg',bbox_inches='tight',pad_inches=0)

plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='fano',hue='size',data=FF_size,ax=ax,size=3,dodge=True)

if save_figures:
    plt.savefig(fig_dir + 'F2B-bottom.svg',bbox_inches='tight',pad_inches=0)

FF_SML = pd.DataFrame(data={'fano':params['fit_fano_SML_nrmd'].values,'size':['SML']*len(params.index),'layer':params['layer'].values})
FF_RF  = pd.DataFrame(data={'fano':params['fit_fano_RF_nrmd'].values,'size':['RF']*len(params.index),'layer':params['layer'].values})
FF_near_SUR_200 = pd.DataFrame(data={'fano':params['fit_fano_near_SUR_200'].values,'size':['SUR']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_400 = pd.DataFrame(data={'fano':params['fit_fano_far_SUR_400'].values,'size':['SUR']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_800 = pd.DataFrame(data={'fano':params['fit_fano_far_SUR_800'].values,'size':['SUR']*len(params.index),'layer':params['layer'].values})

FF_LAR = pd.DataFrame(data={'fano':params['fit_fano_LAR'].values,'size':['LAR']*len(params.index),'layer':params['layer'].values})

FF_size_all = FF_size_all.append(FF_SML)
FF_size_all = FF_size_all.append(FF_RF)
FF_size_all = FF_size_all.append(FF_near_SUR_200)
FF_size_all = FF_size_all.append(FF_far_SUR_400)
FF_size_all = FF_size_all.append(FF_far_SUR_800)
FF_size_all = FF_size_all.append(FF_LAR)

print('\n ANOVA for the main effect of size')
lm = ols('fano ~ C(size)',data=FF_size_all).fit()
table = sm.stats.anova_lm(lm,typ=1)
print(table)

print('\n t-test for fano-factor in SML vs RF across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['fit_fano_SML_nrmd'],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs. near SUR 200 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['fit_fano_near_SUR_200_nrmd'],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs. near SUR 400 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['fit_fano_far_SUR_400_nrmd'],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs. near SUR 800 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['fit_fano_far_SUR_800_nrmd'],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs LAR across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df['fit_fano_LAR_nrmd'],1,nan_policy='omit')))

SEM = params.groupby('layer')[['fit_fano_SML_nrmd',
                               'fit_fano_RF_nrmd',
                               'fit_fano_near_SUR_200_nrmd',                               
                               'fit_fano_far_SUR_400_nrmd',                               
                               'fit_fano_far_SUR_800_nrmd',
                               'fit_fano_LAR_nrmd']].sem()

MEAN = params.groupby('layer')[['fit_fano_SML_nrmd',
                               'fit_fano_RF_nrmd',
                               'fit_fano_near_SUR_200_nrmd',                               
                               'fit_fano_far_SUR_400_nrmd',                               
                               'fit_fano_far_SUR_800_nrmd',
                               'fit_fano_LAR_nrmd']].mean()
print(MEAN)
print(SEM)