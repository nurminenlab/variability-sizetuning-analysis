import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

amplification_DF = pd.read_csv('amplification_DF.csv') 


#plt.figure(1, figsize=(4.5, 1.5))
SEM = amplification_DF.groupby(['layer','qtype_signi'])['bsl'].sem()
amplification_DF.groupby(['layer','qtype_signi'])['bsl'].mean().plot(kind='bar',yerr=SEM)

lm = ols('bsl ~ C(qtype_signi) + C(layer)',data=amplification_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)

SEM = amplification_DF.groupby('layer')[['maxquench_diam','maxamplif_diam']].sem()
amplification_DF.groupby('layer')[['maxquench_diam','maxamplif_diam']].mean().plot(kind='bar',yerr=SEM)

#plt.figure(figsize=(1.481, 1.128))
SEM = amplification_DF.groupby('layer')[['maxquench','maxamplif']].sem()
amplification_DF.groupby('layer')[['maxquench','maxamplif']].mean().plot(kind='bar',yerr=SEM)