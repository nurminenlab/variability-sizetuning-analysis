import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import scipy.stats as sts

amplification_DF = pd.read_csv('amplification_DF.csv') 

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
SEM = amplification_DF[['maxquench_diam','maxamplif_diam']].sem()
amplification_DF[['maxquench_diam','maxamplif_diam']].mean().plot(kind='bar',yerr=SEM)
sts.ttest_ind(amplification_DF['maxquench_diam'],amplification_DF['maxamplif_diam'],nan_policy='omit')

plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='maxquench_diam',data=amplification_DF,color='blue',ax=ax)
sns.swarmplot(x='layer',y='maxamplif_diam',data=amplification_DF,color='orange',ax=ax)
ax.set_yscale('log')

plt.figure()
SEM = amplification_DF.groupby('layer')[['maxquench','maxamplif']].sem()
amplification_DF.groupby('layer')[['maxquench','maxamplif']].mean().plot(kind='bar',yerr=SEM)

plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='maxquench',data=amplification_DF,color='blue',ax=ax)
sns.swarmplot(x='layer',y='maxamplif',data=amplification_DF,color='orange',ax=ax)
#ax.set_yscale('log')