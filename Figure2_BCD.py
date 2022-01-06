import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

amplification_DF = pd.read_csv('amplification_DF.csv') 

plt.figure(2, figsize=(4.5, 1.5))
ax = plt.subplot(1,2,2)
SEM = amplification_DF.groupby(['layer','qtype_signi'])['bsl'].sem()
amplification_DF.groupby(['layer','qtype_signi'])['bsl'].mean().plot(kind='bar',yerr=SEM,ax=ax)

lm = ols('bsl ~ C(qtype_signi) + C(layer)',data=quencher_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)

SEM = amplification_DF.groupby('layer')['maxquench_diam','maxamplif_diam'].sem()
amplification_DF.groupby('layer')['maxquench_diam','maxamplif_diam'].mean().plot(kind='bar',yerr=SEM)

plt.figure(figsize=(1.481, 1.128))
ax = plt.subplot(1,1,1)
SEM = amplification_DF.groupby('layer')['maxquench','maxamplif'].sem()
amplification_DF.groupby('layer')['maxquench','maxamplif'].mean().plot(kind='bar',yerr=SEM,ax=ax)