from statistics import median
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import scipy.stats as sts
import numpy as np

amplification_DF = pd.read_csv('amplification_DF_division.csv')

plt.figure()
SEM = amplification_DF.groupby(['layer','qtype_signi'])['bsl'].sem()
amplification_DF.groupby(['layer','qtype_signi'])['bsl'].mean().plot(kind='bar',yerr=SEM)

plt.figure()
sns.swarmplot(x='layer',y='bsl',data=amplification_DF,hue='qtype_signi')

lm = ols('bsl ~ C(qtype_signi) + C(layer)',data=amplification_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)

plt.figure()
sns.barplot(y='bsl',x='layer',data=amplification_DF)
sns.swarmplot(y='bsl',x='layer',data=amplification_DF)

plt.figure()
sns.barplot(y='bsl',x='qtype_signi',data=amplification_DF)
sns.swarmplot(y='bsl',x='qtype_signi',data=amplification_DF)

plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='maxquench_diam',data=amplification_DF,color='blue',ax=ax)
sns.swarmplot(x='layer',y='maxamplif_diam',data=amplification_DF,color='orange',ax=ax)
ax.set_yscale('log')

LAYERS = amplification_DF.groupby('layer')
ind = 0
SEM_diam = pd.DataFrame(columns=['maxquench_diam','maxamplif_diam'])
for l in LAYERS.groups.keys():
    L = LAYERS.get_group(l)
    maxq_inds = np.random.choice(len(L),size=(len(L),1000),replace=True)
    maxa_inds = np.random.choice(len(L),size=(len(L),1000),replace=True)
    maxq_median_holder = np.nan * np.ones(1000)
    maxa_median_holder = np.nan * np.ones(1000)
    for i in range(maxq_median_holder.shape[0]):
        maxq_median_holder[i] = np.nanmedian(L['maxquench_diam'].values[maxq_inds[:,i]])
        maxa_median_holder[i] = np.nanmedian(L['maxamplif_diam'].values[maxa_inds[:,i]])

    maxq = np.std(maxq_median_holder)
    maxa = np.std(maxa_median_holder)
    tmp_df = pd.DataFrame(data={'maxquench_diam': maxq,'maxamplif_diam': maxa},index=[l])
    SEM_diam = SEM_diam.append(tmp_df,sort=True)
    ind += 1

plt.figure()
ax = plt.subplot(111)
amplification_DF.groupby('layer')[['maxquench_diam','maxamplif_diam']].mean().plot(kind='bar',yerr=SEM_diam,ax=ax)
ax.set_ylim([0,4])


plt.figure()
SEM = amplification_DF.groupby('layer')[['maxquench','maxamplif']].sem()
amplification_DF.groupby('layer')[['maxquench','maxamplif']].mean().plot(kind='bar',yerr=SEM)

plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='maxquench',data=amplification_DF,color='blue',ax=ax)
sns.swarmplot(x='layer',y='maxamplif',data=amplification_DF,color='orange',ax=ax)
#ax.set_yscale('log')