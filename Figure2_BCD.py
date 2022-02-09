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

plt.figure()
ax = plt.subplot(1,2,1)
sns.stripplot(x='layer',y='maxquench_diam',data=amplification_DF,color='blue',ax=ax)
sns.stripplot(x='layer',y='maxamplif_diam',data=amplification_DF,color='orange',ax=ax)
ax.set_yscale('log')

# stimulus diameter relative to RF at maxquench / maxmplif 
RFnormed_maxquench_diam = amplification_DF['maxquench_diam']/amplification_DF['RFdiam']
RFnormed_maxamplif_diam = amplification_DF['maxamplif_diam']/amplification_DF['RFdiam']
amplification_DF.insert(loc=2,column='RFnormed_maxquench_diam',value=RFnormed_maxquench_diam)
amplification_DF.insert(loc=2,column='RFnormed_maxamplif_diam',value=RFnormed_maxamplif_diam)

# stimulus diameter at maxquench / maxmplif 
SG = amplification_DF[amplification_DF['layer']=='SG']
G = amplification_DF[amplification_DF['layer']=='G']
IG = amplification_DF[amplification_DF['layer']=='IG']

maxquench_diam_bootstrap_SG = np.nan * np.ones(n_boots)
maxquench_diam_bootstrap_G  = np.nan * np.ones(n_boots)
maxquench_diam_bootstrap_IG = np.nan * np.ones(n_boots)

maxamplif_diam_bootstrap_SG = np.nan * np.ones(n_boots)
maxamplif_diam_bootstrap_G  = np.nan * np.ones(n_boots)
maxamplif_diam_bootstrap_IG = np.nan * np.ones(n_boots)

maxquench_diam_RFnormed_bootstrap_SG = np.nan * np.ones(n_boots)
maxquench_diam_RFnormed_bootstrap_G  = np.nan * np.ones(n_boots)
maxquench_diam_RFnormed_bootstrap_IG = np.nan * np.ones(n_boots)

maxamplif_diam_RFnormed_bootstrap_SG = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_G  = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_IG = np.nan * np.ones(n_boots)


for i in range(n_boots):
    maxquench_diam_bootstrap_SG[i] = np.nanmedian(np.random.choice(SG['maxquench_diam'],size=len(SG),replace=True))
    maxquench_diam_bootstrap_G[i]  = np.nanmedian(np.random.choice(G['maxquench_diam'],size=len(G),replace=True))
    maxquench_diam_bootstrap_IG[i] = np.nanmedian(np.random.choice(IG['maxquench_diam'],size=len(IG),replace=True))

    maxquench_diam_RFnormed_bootstrap_SG[i] = np.nanmedian(np.random.choice(SG['RFnormed_maxquench_diam'],size=len(SG),replace=True))
    maxquench_diam_RFnormed_bootstrap_G[i]  = np.nanmedian(np.random.choice(G['RFnormed_maxquench_diam'],size=len(G),replace=True))
    maxquench_diam_RFnormed_bootstrap_IG[i] = np.nanmedian(np.random.choice(IG['RFnormed_maxquench_diam'],size=len(IG),replace=True))

    maxamplif_diam_bootstrap_SG[i] = np.nanmedian(np.random.choice(SG['maxamplif_diam'],size=len(SG),replace=True))
    maxamplif_diam_bootstrap_G[i]  = np.nanmedian(np.random.choice(G['maxamplif_diam'],size=len(G),replace=True))
    maxamplif_diam_bootstrap_IG[i] = np.nanmedian(np.random.choice(IG['maxamplif_diam'],size=len(IG),replace=True))
    
    maxamplif_diam_RFnormed_bootstrap_SG[i] = np.nanmedian(np.random.choice(SG['RFnormed_maxamplif_diam'],size=len(SG),replace=True))
    maxamplif_diam_RFnormed_bootstrap_G[i]  = np.nanmedian(np.random.choice(G['RFnormed_maxamplif_diam'],size=len(G),replace=True))
    maxamplif_diam_RFnormed_bootstrap_IG[i] = np.nanmedian(np.random.choice(IG['RFnormed_maxamplif_diam'],size=len(IG),replace=True))

maxquench_diam_SD = np.array([np.nanstd(maxquench_diam_bootstrap_SG),np.nanstd(maxquench_diam_bootstrap_G),np.nanstd(maxquench_diam_bootstrap_IG)])
maxquench_diam = np.array([np.nanmedian(SG['maxquench_diam']),np.nanmedian(G['maxquench_diam']),np.nanmedian(IG['maxquench_diam'])])

RFnormed_maxquench_diam_SD = np.array([np.nanstd(maxquench_diam_RFnormed_bootstrap_SG),np.nanstd(maxquench_diam_RFnormed_bootstrap_G),np.nanstd(maxquench_diam_RFnormed_bootstrap_IG)])
RFnormed_maxquench_diam = np.array([np.nanmedian(SG['RFnormed_maxquench_diam']),np.nanmedian(G['RFnormed_maxquench_diam']),np.nanmedian(IG['RFnormed_maxquench_diam'])])

maxamplif_diam_SD = np.array([np.nanstd(maxquench_diam_bootstrap_SG),np.nanstd(maxquench_diam_bootstrap_G),np.nanstd(maxquench_diam_bootstrap_IG)])
maxamplif_diam = np.array([np.nanmedian(SG['maxamplif_diam']),np.nanmedian(G['maxamplif_diam']),np.nanmedian(IG['maxamplif_diam'])])

RFnormed_maxamplif_diam_SD = np.array([np.nanstd(maxamplif_diam_RFnormed_bootstrap_SG),np.nanstd(maxamplif_diam_RFnormed_bootstrap_G),np.nanstd(maxamplif_diam_RFnormed_bootstrap_IG)])
RFnormed_maxamplif_diam = np.array([np.nanmedian(SG['RFnormed_maxamplif_diam']),np.nanmedian(G['RFnormed_maxamplif_diam']),np.nanmedian(IG['RFnormed_maxamplif_diam'])])

ax = plt.subplot(1,2,2)
ax.bar([0.75,2.75,4.75],maxquench_diam,yerr=maxquench_diam_SD,fc='blue',ec='black',width=0.5)
ax.bar(np.array([0.75,2.75,4.75])+0.5,maxamplif_diam,yerr=maxamplif_diam_SD,fc='orange',ec='black',width=0.5)
ax.set_ylim([0,4.5])
ax.set_xticks([1,3,5])
ax.set_xticklabels(['SG','G','IG'])

plt.figure()
ax = plt.subplot(1,2,1)
sns.stripplot(x='layer',y='RFnormed_maxquench_diam',data=amplification_DF,color='blue',ax=ax)
sns.stripplot(x='layer',y='RFnormed_maxamplif_diam',data=amplification_DF,color='orange',ax=ax)
ax.set_yscale('log')

ax = plt.subplot(1,2,2)
ax.bar([0.75,2.75,4.75],RFnormed_maxquench_diam,yerr=RFnormed_maxquench_diam_SD,fc='blue',ec='black',width=0.5)
ax.bar(np.array([0.75,2.75,4.75])+0.5,RFnormed_maxamplif_diam,yerr=RFnormed_maxamplif_diam_SD,fc='orange',ec='black',width=0.5)
ax.set_ylim([0,9.0])
ax.set_xticks([1,3,5])
ax.set_xticklabels(['SG','G','IG'])

plt.figure()
ax = plt.subplot(1,3,1)
sns.stripplot(x='layer',y='maxquench',data=amplification_DF,color='blue',ax=ax)
sns.stripplot(x='layer',y='maxamplif',data=amplification_DF,color='orange',ax=ax)

ax = plt.subplot(1,3,2)
SEM = amplification_DF.groupby('layer')[['maxquench','maxamplif']].sem()
amplification_DF.groupby('layer')[['maxquench','maxamplif']].mean().plot(kind='bar',yerr=SEM,ax=ax)

plt.figure()
ax = plt.subplot(1,1,1)
SEM = amplification_DF[['maxquench','maxamplif']].sem()
amplification_DF[['maxquench','maxamplif']].mean().plot(kind='bar',yerr=SEM,ax=ax)

DF = amplification_DF
DF.groupby('layer')[['maxamplif']].apply(dalib.outlier, args=(DF['maxamplif'].median(),DF['maxamplif'].mad()))

