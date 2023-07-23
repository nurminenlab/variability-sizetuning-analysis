import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'

var_dict =['0.1/0.2 deg','RF','2RF','4RF','8RF','26 deg']

#params = pd.read_csv(F_dir + 'extracted_params-newselection-Jun2023.csv')
params = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')
# filter dataframe
params = params[['fit_fano_SML', 'fit_fano_RF','fit_fano_near_SUR_200','fit_fano_far_SUR_400', 'fit_fano_far_SUR_800', 'fit_fano_LAR', 'layer']]
params = params.dropna()

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

FF_size = FF_size.append(FF_SML)
FF_size = FF_size.append(FF_RF)
FF_size = FF_size.append(FF_near_SUR_200)
FF_size = FF_size.append(FF_far_SUR_400)
FF_size = FF_size.append(FF_far_SUR_800)
FF_size = FF_size.append(FF_LAR)

funk_switch = 0
functions_list = [sts.mstats.gmean,np.median]
agg_dict_gmean = {key: functions_list[funk_switch] for key in var_dict}

plt.figure()
ax = plt.subplot(111)
# this is a placeholder for the bootstrapped SEM
SEM = params.groupby('layer')[var_dict].sem()

layer_list = ['LSG','L4C','LIG']
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

print('\n t-test for fano-factor in SML vs RF across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df[var_dict[0]],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs. near SUR 200 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df[var_dict[2]],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs. near SUR 400 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df[var_dict[3]],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs. near SUR 800 across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df[var_dict[4]],1,nan_policy='omit')))

print('\n t-test for fano-factor in RF vs LAR across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_1samp(df[var_dict[5]],1,nan_policy='omit')))

print(params.groupby('layer').agg(agg_dict_gmean))
print(SEM)





