import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import scipy.stats as sts
import numpy as np
import sys
sys.path.append('C:/Users/lonurmin/Desktop/code/Analysis/')
import data_analysislib as dalib
import warnings
warnings.filterwarnings('ignore')

save_figures = False

fig_dir  = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/IntermediateFigures/'
dada_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/2-PrecomputedAnalysis/'

amplification_DF = pd.read_csv(dada_dir + 'amplification_DF_division.csv')
n_boots = 3000

# Baseline mean and distribution plots factorized by layer and response type (amplifier vs. quencher)
# ----------------------------------------------------------------------------------------------------------------------
plt.figure()
ax = plt.subplot(111)
SEM = amplification_DF.groupby(['layer','qtype_signi'])['bsl'].sem()
amplification_DF.groupby(['layer','qtype_signi'])['bsl'].mean().plot(kind='bar',yerr=SEM,ax=ax,color='white',edgecolor='black')

SG_mixer = amplification_DF.query('qtype_signi=="mixer" & layer=="SG"')
SG_quencher = amplification_DF.query('qtype_signi=="quencher" & layer=="SG"')
G_mixer = amplification_DF.query('qtype_signi=="mixer" & layer=="G"')
G_quencher = amplification_DF.query('qtype_signi=="quencher" & layer=="G"')
IG_mixer = amplification_DF.query('qtype_signi=="mixer" & layer=="IG"')
IG_quencher = amplification_DF.query('qtype_signi=="quencher" & layer=="IG"')
ax.set_ylim(0,7.5)
if save_figures:
    plt.savefig(fig_dir+'Figure4-D-bsl-means.svg')


print('Baseline difference between quenchers and mixers')
print(amplification_DF.groupby(['layer','qtype_signi'])['bsl'].mean())

print('SEM')
print(SEM)


print('Baseline difference between quenchers and mixers')
print('p-value for SG: ',sts.ttest_ind(SG_mixer['bsl'],SG_quencher['bsl'])[1])
print('p-value G: ',sts.ttest_ind(G_mixer['bsl'],G_quencher['bsl'])[1])
print('p-value IG: ',sts.ttest_ind(IG_mixer['bsl'],IG_quencher['bsl'])[1])

plt.figure()
SZZ = 2
ax = plt.subplot(111)
sns.stripplot(x='layer',y='bsl',data=amplification_DF,hue='qtype_signi',dodge=True,ax=ax)
ax.set_ylim(0,7.5)
if save_figures:
    plt.savefig(fig_dir+'Figure4-D-bsl-distribution.svg')

lm = ols('bsl ~ C(qtype_signi) + C(layer)',data=amplification_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)
print(table)