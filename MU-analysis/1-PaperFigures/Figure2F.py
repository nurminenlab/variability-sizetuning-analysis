import sys
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = True

fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
anal_root = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/2-PrecomputedAnalysis/'

# to generate quencher_DF run generate_quencher_DF.py
quencher_DF = pd.read_csv(anal_root+'quencher_DF.csv')

# bar graph of the proportion of quenchers in each layer
ax = plt.subplot(111)
quencher_DF.groupby(['layer','FF_sup']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar', stacked=True, ax=ax,color=['red','grey','blue'])

if save_figures:
    plt.savefig(fig_dir + 'F2F.svg',bbox_inches='tight',pad_inches=0)

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

FAC = quencher_DF.query('FF_sup == "facilitator"')
SUP = quencher_DF.query('FF_sup == "suppresser"')
print('\n t-test aggregated for across layers for Fano-suppression vs. Fano-facilitation magnitude')
print(sts.ttest_ind(FAC['FF_sup_magn'],SUP['FF_sup_magn'].abs()))