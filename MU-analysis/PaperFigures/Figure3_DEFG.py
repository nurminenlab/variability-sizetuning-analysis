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
dada_dir = 'C:/Users/lonurmin/Desktop/AnalysisScripts/VariabilitySizeTuning/variability-sizetuning-analysis/MU-analysis/'

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

# Baseline mean and distribution plots collapsed acrossed response types and factorized by layer
# ----------------------------------------------------------------------------------------------------------------------
plt.figure()
sns.barplot(y='bsl',x='layer',data=amplification_DF,alpha=0.5)
sns.stripplot(y='bsl',x='layer',data=amplification_DF)
if save_figures:
    plt.savefig(fig_dir+'Figure4-E-bsl-layers-only.svg')

SG_layer = amplification_DF.query('layer=="SG"')
G_layer = amplification_DF.query('layer=="G"')
IG_layer = amplification_DF.query('layer=="IG"')
print('Baseline difference between layers')
print('SG vs G',sts.ttest_ind(SG_layer['bsl'],G_layer['bsl'])[1])
print('SG vs IG',sts.ttest_ind(SG_layer['bsl'],IG_layer['bsl'])[1])
print('G vs IG',sts.ttest_ind(G_layer['bsl'],IG_layer['bsl'])[1])

# Baseline mean and distribution plots collapsed acrossed layer and factorized by response type
# ----------------------------------------------------------------------------------------------------------------------
plt.figure()
sns.barplot(y='bsl',x='qtype_signi',data=amplification_DF,fc='gray')
sns.stripplot(y='bsl',x='qtype_signi',data=amplification_DF,alpha=0.5)

if save_figures:
    plt.savefig(fig_dir+'Figure4-E-bsl-responsetype_only.svg')

mixer = amplification_DF.query('qtype_signi=="mixer"')
quencher = amplification_DF.query('qtype_signi=="quencher"')

print('\nDifference in baseline Fano-factor depending on response type')
print(amplification_DF.groupby('qtype_signi')['bsl'].mean())
print('\nSEM')
print(amplification_DF.groupby('qtype_signi')['bsl'].sem())
print('Baseline difference between mixer vs quencher')
print('p-value',sts.ttest_ind(mixer['bsl'],quencher['bsl'])[1])

# Computations for RF-normalized stimulus sizes at max quench or amplification begin here
# ----------------------------------------------------------------------------------------------------------------------
# stimulus diameter relative to RF at maxquench / maxmplif 
RFnormed_maxquench_diam = amplification_DF['maxquench_diam']/amplification_DF['RFdiam']
RFnormed_maxamplif_diam = amplification_DF['maxamplif_diam']/amplification_DF['RFdiam']
amplification_DF.insert(loc=2,column='RFnormed_maxquench_diam',value=RFnormed_maxquench_diam)
amplification_DF.insert(loc=2,column='RFnormed_maxamplif_diam',value=RFnormed_maxamplif_diam)

# stimulus diameter at maxquench / maxmplif 
SG = amplification_DF.query('layer=="SG"')
G = amplification_DF.query('layer=="G"')
IG = amplification_DF.query('layer=="IG"')

maxquench_diam_RFnormed_bootstrap_SG = np.nan * np.ones(n_boots)
maxquench_diam_RFnormed_bootstrap_G  = np.nan * np.ones(n_boots)
maxquench_diam_RFnormed_bootstrap_IG = np.nan * np.ones(n_boots)

maxamplif_diam_RFnormed_bootstrap_SG = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_G  = np.nan * np.ones(n_boots)
maxamplif_diam_RFnormed_bootstrap_IG = np.nan * np.ones(n_boots)

# Compute RF normed stimulus diameter at maxquench and maxamplif
# ----------------------------------------------------------------------------------------------------------------------
SG = SG[~np.isnan(SG['maxamplif_diam'].values)]
G = G[~np.isnan(G['maxamplif_diam'].values)]
IG = IG[~np.isnan(IG['maxamplif_diam'].values)]

# get indices to outliers
# RF normed stimulus diameter at maxquench
SG['RFnormed_maxquench_diam_outliers'] = SG['RFnormed_maxquench_diam'].apply(dalib.outlier,
                                            args=(SG['RFnormed_maxquench_diam'].median(),SG['RFnormed_maxquench_diam'].mad()))
G['RFnormed_maxquench_diam_outliers'] = G['RFnormed_maxquench_diam'].apply(dalib.outlier,
                                            args=(G['RFnormed_maxquench_diam'].median(),G['RFnormed_maxquench_diam'].mad()))
IG['RFnormed_maxquench_diam_outliers'] = IG['RFnormed_maxquench_diam'].apply(dalib.outlier,
                                            args=(IG['RFnormed_maxquench_diam'].median(),IG['RFnormed_maxquench_diam'].mad()))
# RF normed stimulus diameter at maxamplification
SG['RFnormed_maxamplif_diam_outliers'] = SG['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(SG['RFnormed_maxamplif_diam'].median(),SG['RFnormed_maxamplif_diam'].mad()))
G['RFnormed_maxamplif_diam_outliers'] = G['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(G['RFnormed_maxamplif_diam'].median(),G['RFnormed_maxamplif_diam'].mad()))
IG['RFnormed_maxamplif_diam_outliers'] = IG['RFnormed_maxamplif_diam'].apply(dalib.outlier,
                                            args=(IG['RFnormed_maxamplif_diam'].median(),IG['RFnormed_maxamplif_diam'].mad()))
# amplification magnitude
SG['maxamplif_outliers'] = SG['maxamplif'].apply(dalib.outlier,
                                            args=(SG['maxamplif'].median(),SG['maxamplif'].mad()))
G['maxamplif_outliers'] = G['maxamplif'].apply(dalib.outlier,
                                            args=(G['maxamplif'].median(),G['maxamplif'].mad()))
IG['maxamplif_outliers'] = IG['maxamplif'].apply(dalib.outlier,
                                            args=(IG['maxamplif'].median(),IG['maxamplif'].mad()))
# quenching magnitude
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


# RFnormed stimulus diameter @ maxquench
# bootstrapped SEM
RFnormed_maxquench_diam_SD = np.array([np.nanstd(maxquench_diam_RFnormed_bootstrap_SG),
                                    np.nanstd(maxquench_diam_RFnormed_bootstrap_G),
                                    np.nanstd(maxquench_diam_RFnormed_bootstrap_IG)])
# median
RFnormed_maxquench_diam = np.array([np.nanmedian(SG['RFnormed_maxquench_diam'][SG['RFnormed_maxquench_diam_outliers']==False]),
                                    np.nanmedian(G['RFnormed_maxquench_diam'][G['RFnormed_maxquench_diam_outliers']==False]),
                                    np.nanmedian(IG['RFnormed_maxquench_diam'][IG['RFnormed_maxquench_diam_outliers']==False])])

# RFnormed stimulus diameter @ maxamplif
# bootstrapped SEM
RFnormed_maxamplif_diam_SD = np.array([np.nanstd(maxamplif_diam_RFnormed_bootstrap_SG),
                                        np.nanstd(maxamplif_diam_RFnormed_bootstrap_G),
                                        np.nanstd(maxamplif_diam_RFnormed_bootstrap_IG)])
# median
RFnormed_maxamplif_diam = np.array([np.nanmedian(SG['RFnormed_maxamplif_diam'][SG['RFnormed_maxamplif_diam_outliers']==False]),
                                    np.nanmedian(G['RFnormed_maxamplif_diam'][G['RFnormed_maxamplif_diam_outliers']==False]),
                                    np.nanmedian(IG['RFnormed_maxamplif_diam'][IG['RFnormed_maxquench_diam_outliers']==False])])


# Remove outliers, RF normalized quenching diameter
SG_olRem_diam = pd.DataFrame(SG[SG['RFnormed_maxquench_diam_outliers']==False])
G_olRem_diam  = pd.DataFrame(G[G['RFnormed_maxquench_diam_outliers']==False])
IG_olRem_diam = pd.DataFrame(IG[IG['RFnormed_maxquench_diam_outliers']==False])

# Remove outliers, RF normalized amplification diameter
aSG_olRem_diam = pd.DataFrame(SG[SG['RFnormed_maxamplif_diam_outliers']==False])
aG_olRem_diam  = pd.DataFrame(G[G['RFnormed_maxamplif_diam_outliers']==False])
aIG_olRem_diam = pd.DataFrame(IG[IG['RFnormed_maxamplif_diam_outliers']==False])

# Remove outliers, quenching amplitude 
SG_olRem_ampli = pd.DataFrame(SG[SG['maxquench_outliers']==False])
G_olRem_ampli  = pd.DataFrame(G[G['maxquench_outliers']==False])
IG_olRem_ampli = pd.DataFrame(IG[IG['maxquench_outliers']==False])

# Remove outliers, amplification amplitude 
aSG_olRem_ampli = pd.DataFrame(SG[SG['maxamplif_outliers']==False])
aG_olRem_ampli  = pd.DataFrame(G[G['maxamplif_outliers']==False])
aIG_olRem_ampli = pd.DataFrame(IG[IG['maxamplif_outliers']==False])

# Plot RFnormed stimulus diameters at maxquench and maxamplif factorized by layer
# -----------------------------------------------------------------------------
ms = 4
plt.figure()
ax = plt.subplot(1,2,1)
ax.plot([0.90]*len(SG_olRem_diam),SG_olRem_diam['RFnormed_maxquench_diam'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([1.1]*len(aSG_olRem_diam),aSG_olRem_diam['RFnormed_maxamplif_diam'],'yo',markersize=ms,color='orange',markerfacecolor='white')

ax.plot([1.9]*len(G_olRem_diam),G_olRem_diam['RFnormed_maxquench_diam'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([2.1]*len(aG_olRem_diam),aG_olRem_diam['RFnormed_maxamplif_diam'],'yo',markersize=ms,color='orange',markerfacecolor='white')

ax.plot([2.9]*len(IG_olRem_diam),IG_olRem_diam['RFnormed_maxquench_diam'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([3.1]*len(aIG_olRem_diam),aIG_olRem_diam['RFnormed_maxamplif_diam'],'yo',markersize=ms,color='orange',markerfacecolor='white')

ax.set_xticks([1,2,3])
ax.set_xticklabels(['SG','G','IG'])
ax.set_yscale('log')

ax = plt.subplot(1,2,2)
ax.bar([0.75,2.75,4.75],RFnormed_maxquench_diam,yerr=RFnormed_maxquench_diam_SD,fc='blue',ec='black',width=0.5)
ax.bar(np.array([0.75,2.75,4.75])+0.5,RFnormed_maxamplif_diam,yerr=RFnormed_maxamplif_diam_SD,fc='orange',ec='black',width=0.5)
ax.set_ylim([0,6.0])
ax.set_xticks([1,3,5])
ax.set_xticklabels(['SG','G','IG'])
if save_figures:
    plt.savefig(fig_dir+'Figure4-F-RFnormed-diameter-only.svg')

print('\nRF normed maxquench diameter:')
print(RFnormed_maxquench_diam)
print('SEM')
print(RFnormed_maxquench_diam_SD)

print('\nRF normed maxamplif diameter:')
print(RFnormed_maxamplif_diam)
print('SEM')
print(RFnormed_maxamplif_diam_SD)

# bootstrapped statistics for RFnormed stimulus diameter at maxquench vs maxamplif
# indices to not NaN
SG_quench_notnan = ~np.isnan(SG_olRem_diam['RFnormed_maxquench_diam'].values)
SG_amplif_notnan = ~np.isnan(aSG_olRem_diam['RFnormed_maxamplif_diam'].values)
G_quench_notnan = ~np.isnan(G_olRem_diam['RFnormed_maxquench_diam'].values)
G_amplif_notnan = ~np.isnan(aG_olRem_diam['RFnormed_maxamplif_diam'].values)
IG_quench_notnan = ~np.isnan(IG_olRem_diam['RFnormed_maxquench_diam'].values)
IG_amplif_notnan = ~np.isnan(aIG_olRem_diam['RFnormed_maxamplif_diam'].values)

SG_total_pop = np.concatenate((SG_olRem_diam['RFnormed_maxquench_diam'].values[SG_quench_notnan],
                                aSG_olRem_diam['RFnormed_maxamplif_diam'].values[SG_amplif_notnan]))
G_total_pop  = np.concatenate((G_olRem_diam['RFnormed_maxquench_diam'].values[G_quench_notnan],
                                aG_olRem_diam['RFnormed_maxamplif_diam'].values[G_amplif_notnan]))
IG_total_pop = np.concatenate((IG_olRem_diam['RFnormed_maxquench_diam'].values[IG_quench_notnan],
                                aIG_olRem_diam['RFnormed_maxamplif_diam'].values[IG_amplif_notnan]))

delta_SG_boot = np.nan * np.ones(n_boots)
delta_G_boot = np.nan * np.ones(n_boots)
delta_IG_boot = np.nan * np.ones(n_boots)

for i in range(n_boots):
    SG_boot_pop_quench = np.random.choice(SG_total_pop,size=np.sum(SG_quench_notnan))
    SG_boot_pop_amplif = np.random.choice(SG_total_pop,size=np.sum(SG_amplif_notnan))
    delta_SG_boot[i] = np.nanmedian(SG_boot_pop_quench) - np.nanmedian(SG_boot_pop_amplif)

    G_boot_pop_quench = np.random.choice(G_total_pop,size=np.sum(G_quench_notnan))
    G_boot_pop_amplif = np.random.choice(G_total_pop,size=np.sum(G_amplif_notnan))
    delta_G_boot[i]  = np.nanmedian(G_boot_pop_quench) - np.nanmedian(G_boot_pop_amplif)
    
    IG_boot_pop_quench = np.random.choice(IG_total_pop,size=np.sum(IG_quench_notnan))
    IG_boot_pop_amplif = np.random.choice(IG_total_pop,size=np.sum(IG_amplif_notnan))
    delta_IG_boot[i] = np.nanmedian(IG_boot_pop_quench) - np.nanmedian(IG_boot_pop_amplif)

SG_thr = np.nanmedian(SG_olRem_diam['RFnormed_maxquench_diam'].values) - np.nanmedian(aSG_olRem_diam['RFnormed_maxamplif_diam'].values)
G_thr  = np.nanmedian(G_olRem_diam['RFnormed_maxquench_diam'].values) - np.nanmedian(aG_olRem_diam['RFnormed_maxamplif_diam'].values)    
IG_thr = np.nanmedian(IG_olRem_diam['RFnormed_maxquench_diam'].values) - np.nanmedian(aIG_olRem_diam['RFnormed_maxamplif_diam'].values)

print('\nDifference of median maxquench and maxamplif stimulus diameters:')
print('p-value SG ', np.sum(delta_SG_boot > SG_thr)/n_boots)
print('p-value G ', np.sum(delta_G_boot > G_thr)/n_boots)
print('p-value IG ', np.sum(delta_IG_boot > IG_thr)/n_boots)


# Plotting the maxquench and maxamplif amplitudes factorized by layer
# -----------------------------------------------------------------------------
plt.figure()
SG_olRem_ampli['maxquench-ampli'] = SG_olRem_ampli['maxquench'].apply(lambda x: x * -1)
G_olRem_ampli['maxquench-ampli'] = G_olRem_ampli['maxquench'].apply(lambda x: x * -1)
IG_olRem_ampli['maxquench-ampli'] = IG_olRem_ampli['maxquench'].apply(lambda x: x * -1)

ax = plt.subplot(1,2,1)
ax.plot([0.90]*len(SG_olRem_ampli),SG_olRem_ampli['maxquench-ampli'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([1.1]*len(aSG_olRem_ampli),aSG_olRem_ampli['maxamplif'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.plot([1.9]*len(G_olRem_ampli),G_olRem_ampli['maxquench-ampli'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([2.1]*len(aG_olRem_ampli),aG_olRem_ampli['maxamplif'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.plot([2.9]*len(IG_olRem_ampli),IG_olRem_ampli['maxquench-ampli'],'bo',markersize=ms,markerfacecolor='white')
ax.plot([3.1]*len(aIG_olRem_ampli),aIG_olRem_ampli['maxamplif'],'yo',markersize=ms,color='orange',markerfacecolor='white')
ax.set_xticks([1,2,3])
ax.set_xticklabels(['SG','G','IG'])
ax.set_ylim([0,5.5])


mean_maxquench_ampli = np.array([np.nanmean(SG_olRem_ampli['maxquench-ampli']),
                                np.nanmean(G_olRem_ampli['maxquench-ampli']),
                                np.nanmean(IG_olRem_ampli['maxquench-ampli'])])

SE_maxquench_ampli = np.array([SG_olRem_ampli['maxquench-ampli'].sem(),
                                G_olRem_ampli['maxquench-ampli'].sem(),
                                IG_olRem_ampli['maxquench-ampli'].sem()])

mean_maxamplif_ampli = np.array([np.nanmean(aSG_olRem_ampli['maxamplif']),
                                np.nanmean(aG_olRem_ampli['maxamplif']),
                                np.nanmean(aIG_olRem_ampli['maxamplif'])])

SE_maxamplif_ampli = np.array([aSG_olRem_ampli['maxamplif'].sem(),
                                aG_olRem_ampli['maxamplif'].sem(),
                                aIG_olRem_ampli['maxamplif'].sem()])

ax = plt.subplot(1,2,2)
ax.bar([0.75,2.75,4.75],mean_maxquench_ampli, yerr=SE_maxquench_ampli,fc='blue',ec='black',width=0.5)
ax.bar(np.array([0.75,2.75,4.75])+0.5,mean_maxamplif_ampli,yerr=SE_maxamplif_ampli,fc='orange',ec='black',width=0.5)
ax.set_xticks([1,3,5])
ax.set_xticklabels(['SG','G','IG'])
ax.set_ylim([0,5.5])
if save_figures:
    plt.savefig(fig_dir+'Figure3D-effect-magnitude.svg')

print('\nMean effect magnitude quench')
print(mean_maxquench_ampli)
print('SEM')
print(SE_maxquench_ampli)

print('\nMean effect magnitude amplif')
print(mean_maxamplif_ampli)
print('SEM')
print(SE_maxamplif_ampli)


print('Difference in amplification and quenching magnitude:')
print('SG p-value', sts.ttest_ind(SG_olRem_ampli['maxquench-ampli'],aSG_olRem_ampli['maxamplif'],alternative='greater',nan_policy='omit')[1])
print('G p-value', sts.ttest_ind(G_olRem_ampli['maxquench-ampli'],aG_olRem_ampli['maxamplif'],alternative='greater',nan_policy='omit')[1])
print('IG p-value', sts.ttest_ind(IG_olRem_ampli['maxquench-ampli'],aIG_olRem_ampli['maxamplif'],alternative='greater',nan_policy='omit')[1])
#print('across layers p-value', sts.ttest_ind(ampquench['maxquench-ampli'],ampquench['maxamplif'],nan_policy='omit')[1])


amplification_DF.groupby(['layer','qtype_signi']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum())

# swarmplot RF normalized max quech and amplif stimulus diameters 
plt.figure()
SZZ = 2
frame = [SG_olRem_diam, G_olRem_diam, IG_olRem_diam]
QF = pd.concat(frame)

ax = plt.subplot(2,2,1)
sns.swarmplot(x='layer',y='RFnormed_maxquench_diam',data=QF,color='blue',size=SZZ)
ax.set_ylim(0.05,70)
ax.set_yscale('log')

ax = plt.subplot(2,2,2)
QF.groupby('layer')['RFnormed_maxquench_diam'].median().plot(kind='bar',yerr=RFnormed_maxquench_diam_SD, color='white',ec='black')
ax.set_ylim(0.05,70)
ax.set_yscale('log')

frame = [aSG_olRem_diam, aG_olRem_diam, aIG_olRem_diam]
aQF = pd.concat(frame)

ax = plt.subplot(2,2,3)
sns.swarmplot(x='layer',y='RFnormed_maxamplif_diam',data=aQF,color='orange',size=SZZ)
ax.set_ylim(0.05,70)
ax.set_yscale('log')

ax = plt.subplot(2,2,4)
aQF.groupby('layer')['RFnormed_maxamplif_diam'].median().plot(kind='bar',yerr=RFnormed_maxamplif_diam_SD, color='white',ec='black')
ax.set_ylim(0.05,70)
ax.set_yscale('log')

if save_figures:
    plt.savefig(fig_dir+'Figure_3C-amp_quench_diameters.svg')

print('Difference in baseline FR between mixers and quenchers')
amplification_DF['bsl_FR_new'] = amplification_DF['bsl_FR'].apply(lambda x: x/0.4)
amplification_DF.groupby('qtype_signi')['bsl_FR_new'].mean()
print('SEM')
amplification_DF.groupby('qtype_signi')['bsl_FR_new'].sem()

mixers = amplification_DF[amplification_DF['qtype_signi']=='mixer']
quenchers = amplification_DF[amplification_DF['qtype_signi']=='quencher']

print(sts.ttest_ind(mixers['bsl_FR'],quenchers['bsl_FR_new'],nan_policy='omit'))

