import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
params = pd.read_csv(F_dir + 'extracted_params-Dec-2021.csv')

FF_size = pd.DataFrame(columns=['fano','size','layer'])
FF_size_all = pd.DataFrame(columns=['fano','size','layer'])



FF_RF = pd.DataFrame(data={'fano':params['fit_fano_RF'].values,'size':['RF']*len(params.index),'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'fano':params['fit_fano_LAR'].values,'size':['LAR']*len(params.index),'layer':params['layer'].values})

FF_size = FF_size.append(FF_RF)
FF_size = FF_size.append(FF_LAR)

plt.figure()
ax = plt.subplot(111)
SEM = params.groupby('layer')[['fit_fano_SML','fit_fano_RF','fit_fano_SUR','fit_fano_LAR']].sem()
params.groupby('layer')[['fit_fano_SML','fit_fano_RF','fit_fano_SUR','fit_fano_LAR']].mean().plot(ax=ax,kind='bar',yerr=SEM)

plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='fano',hue='size',data=FF_size,ax=ax,size=3,dodge=True)

FF_SML = pd.DataFrame(data={'fano':params['fit_fano_SML'].values,'size':['SML']*len(params.index),'layer':params['layer'].values})
FF_RF  = pd.DataFrame(data={'fano':params['fit_fano_RF'].values,'size':['RF']*len(params.index),'layer':params['layer'].values})
FF_SUR = pd.DataFrame(data={'fano':params['fit_fano_SUR'].values,'size':['SUR']*len(params.index),'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'fano':params['fit_fano_LAR'].values,'size':['LAR']*len(params.index),'layer':params['layer'].values})

FF_size_all = FF_size_all.append(FF_SML)
FF_size_all = FF_size_all.append(FF_RF)
FF_size_all = FF_size_all.append(FF_SUR)
FF_size_all = FF_size_all.append(FF_LAR)

print('\n ANOVA for the main effect of size')
lm = ols('fano ~ C(size)',data=FF_size_all).fit()
table = sm.stats.anova_lm(lm,typ=1)
print(table)

print('\n t-test for fano-factor in SML vs RF across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_ind(df['fit_fano_SML'],df['fit_fano_RF'],nan_policy='omit')))

print('\n t-test for fano-factor in RF vs LAR across layers')
print(params.groupby('layer').apply(lambda df: sts.ttest_ind(df['fit_fano_LAR'],df['fit_fano_RF'],nan_policy='omit')))

sts.f_oneway()
