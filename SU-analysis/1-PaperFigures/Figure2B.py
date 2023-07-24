import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'

params = pd.read_csv(F_dir + 'SU-extracted_params-Jul2023.csv')

var_dict =['0.1/0.2 deg','RF','2RF','4RF','8RF','26 deg']

# we clean up units without much fano factor tuning
SG_units_to_remove = [1,7,14,51,53,58,80]
IG_units_to_remove = [20,31,32,34,46,77,81]


idx_to_remove = []
for unit in SG_units_to_remove:
    idx_to_remove.append(params[params['unit'] == unit].index[0])

for unit in IG_units_to_remove:
    idx_to_remove.append(params[params['unit'] == unit].index[0])

params.drop(idx_to_remove,axis=0,inplace=True)
params.drop(params[params['layer'] == 'L4C'].index,axis=0,inplace=True) # just one L4C unit

FF_size = pd.DataFrame(columns=['fano','size','layer'])
FF_size_all = pd.DataFrame(columns=['fano','size','layer'])

params[var_dict[0]] = params['fit_fano_SML'] / params['fit_fano_RF']
params[var_dict[1]] = params['fit_fano_RF'] / params['fit_fano_RF']
params[var_dict[2]] = params['fit_fano_near_SUR_200'] / params['fit_fano_RF']
params[var_dict[3]] = params['fit_fano_far_SUR_400'] / params['fit_fano_RF']
params[var_dict[4]] = params['fit_fano_far_SUR_800'] / params['fit_fano_RF']
params[var_dict[5]] = params['fit_fano_LAR'] / params['fit_fano_RF']


FF_SML = pd.DataFrame(data={'fano':params[var_dict[0]].values,'size':['0.1/0.2 deg']*len(params.index),'layer':params['layer'].values})
FF_RF = pd.DataFrame(data={'fano':params[var_dict[1]].values,'size':['RF']*len(params.index),'layer':params['layer'].values})
FF_near_SUR_200 = pd.DataFrame(data={'fano':params[var_dict[2]].values,'size':['2RF']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_400 = pd.DataFrame(data={'fano':params[var_dict[3]].values,'size':['4RF']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_800 = pd.DataFrame(data={'fano':params[var_dict[4]].values,'size':['8RF']*len(params.index),'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'fano':params[var_dict[5]].values,'size':['26 deg']*len(params.index),'layer':params['layer'].values})

FF_size = pd.concat([FF_size,FF_SML])
FF_size = pd.concat([FF_size,FF_RF])
FF_size = pd.concat([FF_size,FF_near_SUR_200])
FF_size = pd.concat([FF_size,FF_far_SUR_400])
FF_size = pd.concat([FF_size,FF_far_SUR_800])
FF_size = pd.concat([FF_size,FF_LAR])

funk_switch = 0
functions_list = [sts.mstats.gmean,np.median]
agg_dict_gmean = {key: functions_list[funk_switch] for key in var_dict}

plt.figure()

ax = plt.subplot(111)
# this is a placeholder for the bootstrapped SEM
SEM = params.groupby('layer')[var_dict].sem()

layer_list = ['LSG','LIG']
p = params.groupby('layer')
# layer loop 
for l in layer_list:
    for cond in SEM.columns:
        if cond == var_dict[1]:
            SEM[cond][l] = 0
        else:
            SEM[cond][l] = sts.bootstrap((p[cond].get_group(l).values,),functions_list[funk_switch]).standard_error

params.groupby('layer').agg(agg_dict_gmean).plot(ax=ax,kind='bar',yerr=SEM)
ax.set_ylabel('Normalized fano-factor')    
if save_figures:
    plt.savefig(fig_dir + 'F2B-top.svg',bbox_inches='tight',pad_inches=0)


plt.figure()

ax = plt.subplot(111)
sns.swarmplot(x='layer',y='fano',hue='size',data=FF_size,ax=ax,size=3,dodge=True)
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'gray'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}}
sns.boxplot(x='layer',y='fano',hue='size',data=FF_size,ax=ax,showfliers=False,**PROPS)
ax.set_ylabel('Normalized fano-factor')
ax.legend('')
if save_figures:
    plt.savefig(fig_dir + 'F2B-bottom.svg',bbox_inches='tight',pad_inches=0)

FF_SML = pd.DataFrame(data={'fano':params[var_dict[0]].values,'size':[var_dict[0]]*len(params.index),'layer':params['layer'].values})
FF_RF  = pd.DataFrame(data={'fano':params[var_dict[1]].values,'size':[var_dict[1]]*len(params.index),'layer':params['layer'].values})
FF_near_SUR_200 = pd.DataFrame(data={'fano':params[var_dict[2]].values,'size':[var_dict[2]]*len(params.index),'layer':params['layer'].values})
FF_far_SUR_400 = pd.DataFrame(data={'fano':params[var_dict[3]].values,'size':[var_dict[4]]*len(params.index),'layer':params['layer'].values})
FF_far_SUR_800 = pd.DataFrame(data={'fano':params[var_dict[4]].values,'size':[var_dict[5]]*len(params.index),'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'fano':params[var_dict[5]].values,'size':[var_dict[5]]*len(params.index),'layer':params['layer'].values})

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

var_dict_log =['0.1/0.2 deg log','RF log','2RF log','4RF log','8RF log','26 deg log']




print('\n bootstrap test for fano-factor in SML vs RF across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(np.log(df[var_dict[0]].values),0,nan_policy='omit',alternative='greater')))

print('\n t-test for fano-factor in RF vs. near SUR 200 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(np.log(df[var_dict[2]].values),0,nan_policy='omit',alternative='less')))

print('\n t-test for fano-factor in RF vs. near SUR 400 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(np.log(df[var_dict[3]].values),0,nan_policy='omit',alternative='less')))

print('\n t-test for fano-factor in RF vs. near SUR 800 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(np.log(df[var_dict[4]].values),0,nan_policy='omit',alternative='less')))

print('\n t-test for fano-factor in RF vs LAR across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(np.log(df[var_dict[5]].values),0,nan_policy='omit',alternative='less')))

print(params.groupby('layer').agg(agg_dict_gmean))
print(SEM)