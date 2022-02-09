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

amplification_DF = pd.read_csv('amplification_DF_division.csv')
n_boots = 1000

plt.figure()
SEM = amplification_DF.groupby(['layer','qtype_signi'])['bsl'].sem()
amplification_DF.groupby(['layer','qtype_signi'])['bsl'].mean().plot(kind='bar',yerr=SEM)

plt.figure()
sns.stripplot(x='layer',y='bsl',data=amplification_DF,hue='qtype_signi',dodge=True)

lm = ols('bsl ~ C(qtype_signi) + C(layer)',data=amplification_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)

plt.figure()
sns.barplot(y='bsl',x='layer',data=amplification_DF,alpha=0.5)
sns.stripplot(y='bsl',x='layer',data=amplification_DF)

plt.figure()
sns.barplot(y='bsl',x='qtype_signi',data=amplification_DF,fc='gray')
sns.stripplot(y='bsl',x='qtype_signi',data=amplification_DF,alpha=0.5)

# stimulus diameter relative to RF at maxquench / maxmplif 
RFnormed_maxquench_diam = amplification_DF['maxquench_diam']/amplification_DF['RFdiam']
RFnormed_maxamplif_diam = amplification_DF['maxamplif_diam']/amplification_DF['RFdiam']
amplification_DF.insert(loc=2,column='RFnormed_maxquench_diam',value=RFnormed_maxquench_diam)
amplification_DF.insert(loc=2,column='RFnormed_maxamplif_diam',value=RFnormed_maxamplif_diam)

# stimulus diameter at maxquench / maxmplif 
SG = amplification_DF[amplification_DF['layer']=='SG']
G = amplification_DF[amplification_DF['layer']=='G']
IG = amplification_DF[amplification_DF['layer']=='IG']

maxquench_diam_RFnormed_bootstrap_SG = np.nan * np.ones(n_boots)
maxquench_diam_RFnormed_bootstrap_G  = np.nan * np.ones(n_boots)
maxquench_diam_RFnormed_bootstrap_IG = np.nan * np.ones(n_boots)

maxamplif_diam_RFnormed_bootstrap_SG = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_G  = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_IG = np.nan * np.ones(n_boots)

# get indices to outliers
SG['RFnormed_maxquench_diam_outliers'] = SG['RFnormed_maxquench_diam'].apply(dalib.outlier,
                                            args=(SG['RFnormed_maxquench_diam'].median(),SG['RFnormed_maxquench_diam'].mad()))
G['RFnormed_maxquench_diam_outliers'] = G['RFnormed_maxquench_diam'].apply(dalib.outlier,
                                            args=(G['RFnormed_maxquench_diam'].median(),G['RFnormed_maxquench_diam'].mad()))
IG['RFnormed_maxquench_diam_outliers'] = IG['RFnormed_maxquench_diam'].apply(dalib.outlier,
                                            args=(IG['RFnormed_maxquench_diam'].median(),IG['RFnormed_maxquench_diam'].mad()))

SG['RFnormed_maxamplif_diam_outliers'] = SG['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(SG['RFnormed_maxamplif_diam'].median(),SG['RFnormed_maxamplif_diam'].mad()))
G['RFnormed_maxamplif_diam_outliers'] = G['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(G['RFnormed_maxamplif_diam'].median(),G['RFnormed_maxamplif_diam'].mad()))
IG['RFnormed_maxamplif_diam_outliers'] = IG['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(IG['RFnormed_maxamplif_diam'].median(),IG['RFnormed_maxamplif_diam'].mad()))

SG['maxamplif_outliers'] = SG['maxamplif'].apply(dalib.outlier,
                                            args=(SG['maxamplif'].median(),SG['maxamplif'].mad()))
G['maxamplif_outliers'] = G['maxamplif'].apply(dalib.outlier,
                                            args=(G['maxamplif'].median(),G['maxamplif'].mad()))
IG['maxamplif_outliers'] = IG['maxamplif'].apply(dalib.outlier,
                                            args=(IG['maxamplif'].median(),IG['maxamplif'].mad()))

SG['maxquench_outliers'] = SG['maxquench'].apply(dalib.outlier,
                                            args=(SG['maxquench'].median(),SG['maxquench'].mad()))
G['maxquench_outliers'] = G['maxquench'].apply(dalib.outlier,
                                            args=(G['maxquench'].median(),G['maxquench'].mad()))
IG['maxquench_outliers'] = IG['maxquench'].apply(dalib.outlier,
                                            args=(IG['maxquench'].median(),IG['maxquench'].mad()))

for i in range(n_boots):
    maxquench_diam_RFnormed_bootstrap_SG[i] = np.nanmedian(np.random.choice(SG['RFnormed_maxquench_diam'][SG['RFnormed_maxquench_diam_outliers']==False],
                                                            size=np.sum(~SG['RFnormed_maxquench_diam_outliers']),replace=True))
    maxquench_diam_RFnormed_bootstrap_G[i]  = np.nanmedian(np.random.choice(G['RFnormed_maxquench_diam'][G['RFnormed_maxquench_diam_outliers']==False],
                                                            size=np.sum(~G['RFnormed_maxquench_diam_outliers']),replace=True))
    maxquench_diam_RFnormed_bootstrap_IG[i] = np.nanmedian(np.random.choice(IG['RFnormed_maxquench_diam'][IG['RFnormed_maxquench_diam_outliers']==False],
                                                            size=np.sum(~IG['RFnormed_maxquench_diam_outliers']),replace=True))
    
    maxamplif_diam_RFnormed_bootstrap_SG[i] = np.nanmedian(np.random.choice(SG['RFnormed_maxamplif_diam'][SG['RFnormed_maxamplif_diam_outliers']==False],
                                                            size=np.sum(~SG['RFnormed_maxamplif_diam_outliers']),replace=True))
    maxamplif_diam_RFnormed_bootstrap_G[i]  = np.nanmedian(np.random.choice(G['RFnormed_maxamplif_diam'][G['RFnormed_maxamplif_diam_outliers']==False],
                                                            size=np.sum(~G['RFnormed_maxamplif_diam_outliers']),replace=True))
    maxamplif_diam_RFnormed_bootstrap_IG[i] = np.nanmedian(np.random.choice(IG['RFnormed_maxamplif_diam'][IG['RFnormed_maxamplif_diam_outliers']==False],
                                                            size=np.sum(~IG['RFnormed_maxamplif_diam_outliers']),replace=True))



RFnormed_maxquench_diam_SD = np.array([np.nanstd(maxquench_diam_RFnormed_bootstrap_SG),
                                    np.nanstd(maxquench_diam_RFnormed_bootstrap_G),
                                    np.nanstd(maxquench_diam_RFnormed_bootstrap_IG)])

RFnormed_maxquench_diam = np.array([np.nanmedian(SG['RFnormed_maxquench_diam'][SG['RFnormed_maxquench_diam_outliers']==False]),
                                    np.nanmedian(G['RFnormed_maxquench_diam'][G['RFnormed_maxquench_diam_outliers']==False]),
                                    np.nanmedian(IG['RFnormed_maxquench_diam'][IG['RFnormed_maxquench_diam_outliers']==False])])

RFnormed_maxamplif_diam_SD = np.array([np.nanstd(maxamplif_diam_RFnormed_bootstrap_SG),np.nanstd(maxamplif_diam_RFnormed_bootstrap_G),np.nanstd(maxamplif_diam_RFnormed_bootstrap_IG)])
RFnormed_maxamplif_diam = np.array([np.nanmedian(SG['RFnormed_maxamplif_diam']),np.nanmedian(G['RFnormed_maxamplif_diam']),np.nanmedian(IG['RFnormed_maxamplif_diam'])])

SG_olRem = pd.DataFrame(SG[SG['RFnormed_maxquench_diam_outliers']==False])
G_olRem  = pd.DataFrame(G[G['RFnormed_maxquench_diam_outliers']==False])
IG_olRem = pd.DataFrame(IG[IG['RFnormed_maxquench_diam_outliers']==False])
RFnormed = SG_olRem.append(G_olRem).append(IG_olRem)

SG_olRem = pd.DataFrame(SG[SG['maxquench_outliers']==False])
G_olRem  = pd.DataFrame(G[G['maxquench_outliers']==False])
IG_olRem = pd.DataFrame(IG[IG['maxquench_outliers']==False])
aSG_olRem = pd.DataFrame(SG[SG['maxamplif_outliers']==False])
aG_olRem  = pd.DataFrame(G[G['maxamplif_outliers']==False])
aIG_olRem = pd.DataFrame(IG[IG['maxamplif_outliers']==False])
ampquench = SG_olRem.append(G_olRem).append(IG_olRem).append(aSG_olRem).append(aG_olRem).append(aIG_olRem)

plt.figure()
ax = plt.subplot(1,2,1)
sns.stripplot(x='layer',y='RFnormed_maxquench_diam',data=RFnormed,color='blue',ax=ax,dodge=True)
sns.stripplot(x='layer',y='RFnormed_maxamplif_diam',data=RFnormed,color='orange',ax=ax,dodge=True)
ax.set_yscale('log')

ax = plt.subplot(1,2,2)
ax.bar([0.75,2.75,4.75],RFnormed_maxquench_diam,yerr=RFnormed_maxquench_diam_SD,fc='blue',ec='black',width=0.5)
ax.bar(np.array([0.75,2.75,4.75])+0.5,RFnormed_maxamplif_diam,yerr=RFnormed_maxamplif_diam_SD,fc='orange',ec='black',width=0.5)
ax.set_ylim([0,9.0])
ax.set_xticks([1,3,5])
ax.set_xticklabels(['SG','G','IG'])

plt.figure()
ax = plt.subplot(1,3,1)
sns.stripplot(x='layer',y='maxquench',data=ampquench,color='blue',ax=ax)
sns.stripplot(x='layer',y='maxamplif',data=ampquench,color='orange',ax=ax)

ax = plt.subplot(1,3,2)
SEM = ampquench.groupby('layer')[['maxquench','maxamplif']].sem()
ampquench.groupby('layer')[['maxquench','maxamplif']].mean().plot(kind='bar',yerr=SEM,ax=ax)

plt.figure()
ax = plt.subplot(1,1,1)
SEM = ampquench[['maxquench','maxamplif']].sem()
ampquench[['maxquench','maxamplif']].mean().plot(kind='bar',yerr=SEM,ax=ax)
