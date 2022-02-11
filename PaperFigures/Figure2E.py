import sys
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm

fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'

# to generate quencher_DF run generate_quencher_DF.py
quencher_DF = pd.read_csv('quencher_DF.csv')

# bar graph of the proportion of quenchers in each layer
ax = plt.subplot(111)
quencher_DF.groupby(['layer','FF_sup']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar', stacked=True, ax=ax,color=['red','grey','blue'])
plt.savefig(fig_dir + 'F2E.svg',bbox_inches='tight',pad_inches=0)
# the proportion of quenchers 
print('\n The proportion of quenchers:')
print(quencher_DF.groupby(['FF_sup']).size() / len(quencher_DF))
# the number of quenchers 
print('\n The number of quenchers:')
print(quencher_DF.groupby(['FF_sup']).size())

# print out the percentage of quenchers in each layer
print(quencher_DF.groupby(['layer','FF_sup']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()))

# main effect of FF_sup
print('\n The main effect of FF_sup:')
lm = ols('SI ~ C(FF_sup)',data=quencher_DF).fit()
print(sm.stats.anova_lm(lm,typ=1))

# means of SI for each FF_sup class
print('\n The means of SI for each FF_sup class:')
print(quencher_DF.groupby('FF_sup')['SI'].mean())
print(quencher_DF.groupby('FF_sup')['SI'].sem())

print('\n Mean change of fano factor for each FF_sup class in different layers')
print(quencher_DF.groupby(['FF_sup','layer'])['FF_sup_magn'].mean())
print('\n SEM ')
print(quencher_DF.groupby(['FF_sup','layer'])['FF_sup_magn'].sem())