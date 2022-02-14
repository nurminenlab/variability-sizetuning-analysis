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

amplification_DF = pd.read_csv('../amplification_DF_division.csv')
n_boots = 1000

plt.figure()
SEM = amplification_DF.groupby(['layer','qtype_signi'])['bsl'].sem()
amplification_DF.groupby(['layer','qtype_signi'])['bsl'].mean().plot(kind='bar',yerr=SEM)
SG_mixer = amplification_DF.query('qtype_signi=="mixer" & layer=="SG"')
SG_quencher = amplification_DF.query('qtype_signi=="quencher" & layer=="SG"')
G_mixer = amplification_DF.query('qtype_signi=="mixer" & layer=="G"')
G_quencher = amplification_DF.query('qtype_signi=="quencher" & layer=="G"')
IG_mixer = amplification_DF.query('qtype_signi=="mixer" & layer=="IG"')
IG_quencher = amplification_DF.query('qtype_signi=="quencher" & layer=="IG"')

print('Baseline difference between quenchers and mixers')
print('p-value for SG: ',sts.ttest_ind(SG_mixer['bsl'],SG_quencher['bsl'])[1])
print('p-value G: ',sts.ttest_ind(G_mixer['bsl'],G_quencher['bsl'])[1])
print('p-value IG: ',sts.ttest_ind(IG_mixer['bsl'],IG_quencher['bsl'])[1])

plt.figure()
sns.stripplot(x='layer',y='bsl',data=amplification_DF,hue='qtype_signi',dodge=True)

lm = ols('bsl ~ C(qtype_signi) + C(layer)',data=amplification_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)
print(table)

plt.figure()
sns.barplot(y='bsl',x='layer',data=amplification_DF,alpha=0.5)
sns.stripplot(y='bsl',x='layer',data=amplification_DF)

SG_layer = amplification_DF.query('layer=="SG"')
G_layer = amplification_DF.query('layer=="G"')
IG_layer = amplification_DF.query('layer=="IG"')
print('Baseline difference between layers')
print('SG vs G',sts.ttest_ind(SG_layer['bsl'],G_layer['bsl'])[1])
print('SG vs IG',sts.ttest_ind(SG_layer['bsl'],IG_layer['bsl'])[1])
print('G vs IG',sts.ttest_ind(G_layer['bsl'],IG_layer['bsl'])[1])

plt.figure()
sns.barplot(y='bsl',x='qtype_signi',data=amplification_DF,fc='gray')
sns.stripplot(y='bsl',x='qtype_signi',data=amplification_DF,alpha=0.5)
mixer = amplification_DF.query('qtype_signi=="mixer"')
quencher = amplification_DF.query('qtype_signi=="quencher"')
print('Baseline difference between mixer vs quencher')
print('p-value',sts.ttest_ind(mixer['bsl'],quencher['bsl'])[1])


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


SG = RFnormed[RFnormed['layer']=='SG']
G = RFnormed[RFnormed['layer']=='G']
IG = RFnormed[RFnormed['layer']=='IG']

ms = 4
plt.figure()
ax = plt.subplot(1,2,1)
ax.plot([0.90]*len(SG),SG['RFnormed_maxquench_diam'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([1.1]*len(SG),SG['RFnormed_maxamplif_diam'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.plot([1.9]*len(G),G['RFnormed_maxquench_diam'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([2.1]*len(G),G['RFnormed_maxamplif_diam'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.plot([2.9]*len(IG),IG['RFnormed_maxquench_diam'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([3.1]*len(IG),IG['RFnormed_maxamplif_diam'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.set_xticks([1,2,3])
ax.set_xticklabels(['SG','G','IG'])
ax.set_yscale('log')

ax = plt.subplot(1,2,2)
ax.bar([0.75,2.75,4.75],RFnormed_maxquench_diam,yerr=RFnormed_maxquench_diam_SD,fc='blue',ec='black',width=0.5)
ax.bar(np.array([0.75,2.75,4.75])+0.5,RFnormed_maxamplif_diam,yerr=RFnormed_maxamplif_diam_SD,fc='orange',ec='black',width=0.5)
ax.set_ylim([0,9.0])
ax.set_xticks([1,3,5])
ax.set_xticklabels(['SG','G','IG'])

ampquench['maxquench-ampli'] = ampquench['maxquench'].apply(lambda x: x * -1)
SG = ampquench[ampquench['layer']=='SG']
G = ampquench[ampquench['layer']=='G']
IG = ampquench[ampquench['layer']=='IG']


plt.figure()
ax = plt.subplot(1,2,1)
ax.plot([0.90]*len(SG),SG['maxquench-ampli'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([1.1]*len(SG),SG['maxamplif'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.plot([1.9]*len(G),G['maxquench-ampli'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([2.1]*len(G),G['maxamplif'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.plot([2.9]*len(IG),IG['maxquench-ampli'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([3.1]*len(IG),IG['maxamplif'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.set_xticks([1,2,3])
ax.set_xticklabels(['SG','G','IG'])

print('Difference in amplification and quenching magnitude:')
print('SG p-value', sts.ttest_ind(SG['maxquench-ampli'],SG['maxamplif'],nan_policy='omit')[1])
print('G p-value', sts.ttest_ind(G['maxquench-ampli'],G['maxamplif'],nan_policy='omit')[1])
print('IG p-value', sts.ttest_ind(IG['maxquench-ampli'],IG['maxamplif'],nan_policy='omit')[1])
print('across layers p-value', sts.ttest_ind(ampquench['maxquench-ampli'],ampquench['maxamplif'],nan_policy='omit')[1])

ax = plt.subplot(1,2,2)
SEM = ampquench.groupby('layer')[['maxquench-ampli','maxamplif']].sem()
ampquench.groupby('layer')[['maxquench-ampli','maxamplif']].mean().plot(kind='bar',yerr=SEM,ax=ax)

plt.figure()
ax = plt.subplot(1,1,1)
SEM = ampquench[['maxquench-ampli','maxamplif']].sem()
ampquench[['maxquench-ampli','maxamplif']].mean().plot(kind='bar',yerr=SEM,ax=ax)

# stimulus diameter at maxquench / maxmplif 
SG = amplification_DF[amplification_DF['layer']=='SG']
G = amplification_DF[amplification_DF['layer']=='G']
IG = amplification_DF[amplification_DF['layer']=='IG']

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

SG_maxquench_olRem = SG['RFnormed_maxquench_diam'][SG['RFnormed_maxquench_diam_outliers']==False].values
SG_maxamplif_olRem = SG['RFnormed_maxamplif_diam'][SG['RFnormed_maxamplif_diam_outliers']==False].values
SG_total_pop = np.concatenate((SG_maxquench_olRem,SG_maxamplif_olRem))

G_maxquench_olRem = G['RFnormed_maxquench_diam'][G['RFnormed_maxquench_diam_outliers']==False]
G_maxamplif_olRem = G['RFnormed_maxamplif_diam'][G['RFnormed_maxamplif_diam_outliers']==False]
G_total_pop = np.concatenate((G_maxquench_olRem,G_maxamplif_olRem))

IG_maxquench_olRem = IG['RFnormed_maxquench_diam'][IG['RFnormed_maxquench_diam_outliers']==False]
IG_maxamplif_olRem = IG['RFnormed_maxamplif_diam'][IG['RFnormed_maxamplif_diam_outliers']==False]
IG_total_pop = np.concatenate((IG_maxquench_olRem,IG_maxamplif_olRem))

delta_SG_boot = np.nan * np.ones(n_boots)
delta_G_boot = np.nan * np.ones(n_boots)
delta_IG_boot = np.nan * np.ones(n_boots)
for i in range(n_boots):
    delta_SG_boot[i] = np.nanmedian(np.random.choice(SG_total_pop,size=len(SG_maxquench_olRem),replace=True)) - np.nanmedian(np.random.choice(SG_total_pop,size=len(SG_maxamplif_olRem),replace=True))
    delta_G_boot[i]  = np.nanmedian(np.random.choice(G_total_pop,size=len(G_maxquench_olRem),replace=True)) - np.nanmedian(np.random.choice(G_total_pop,size=len(G_maxamplif_olRem),replace=True))
    delta_IG_boot[i] = np.nanmedian(np.random.choice(IG_total_pop,size=len(IG_maxquench_olRem),replace=True)) - np.nanmedian(np.random.choice(IG_total_pop,size=len(IG_maxamplif_olRem),replace=True))

SG_thr = np.nanmedian(SG_maxquench_olRem) - np.nanmedian(SG_maxamplif_olRem)
G_thr  = np.nanmedian(G_maxquench_olRem) - np.nanmedian(G_maxamplif_olRem)    
IG_thr = np.nanmedian(IG_maxquench_olRem) - np.nanmedian(IG_maxamplif_olRem)

print('Difference of median maxquench and maxamplif stimulus diameters:')
print('p-value SG ', np.sum(delta_SG_boot > SG_thr)/n_boots)
print('p-value G ', np.sum(delta_G_boot > G_thr)/n_boots)
print('p-value IG ', np.sum(delta_IG_boot > IG_thr)/n_boots)