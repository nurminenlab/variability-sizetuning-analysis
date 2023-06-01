# import packages
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import scipy.stats as sts
import numpy as np

F_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/'

amplification_DF = pd.read_csv(F_dir + 'amplification_DF_division.csv')

plt.figure()
SZZ = 2
amplification_DF['maxquench'] = amplification_DF['maxquench']*-1

ax = plt.subplot(2,2,1)
sns.swarmplot(x='layer',y='maxquench',data=amplification_DF,color='blue',ax=ax,size=SZZ)
ax.set_ylim(0.1, 10)

ax = plt.subplot(2,2,2)
SEM = amplification_DF.groupby('layer')[['maxquench']].sem()
amplification_DF.groupby('layer')[['maxquench']].mean().plot(kind='bar',yerr=SEM,ax=ax,color='white',edgecolor='black')
ax.set_ylim(0, 10)

print('maxquench mean')
print(amplification_DF.groupby('layer')[['maxquench']].mean())
print('maxquench SEM')
print(SEM)

ax = plt.subplot(2,2,3)
sns.swarmplot(x='layer',y='maxamplif',data=amplification_DF,color='orange',ax=ax,size=SZZ)
ax.set_ylim(0, 10)

ax = plt.subplot(2,2,4)
SEM = amplification_DF.groupby('layer')[['maxamplif']].sem()
amplification_DF.groupby('layer')[['maxamplif']].mean().plot(kind='bar',yerr=SEM,ax=ax,color='white',edgecolor='black')
ax.set_ylim(0, 10)

print('maxamplif mean')
print(amplification_DF.groupby('layer')[['maxamplif']].mean())
print('maxamplif SEM')
print(SEM)

# just to make sure not to use the transformed data elsewhere
amplification_DF['maxquench'] = amplification_DF['maxquench']*-1

lm = ols('maxquench ~ C(layer)',data=amplification_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)
print('\n ANOVA over layers for maximum quenching amplitude:')
print(table)

lm = ols('maxamplif ~ C(layer)',data=amplification_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)
print('\n ANOVA over layers for maximum amplification amplitude:')
print(table)

