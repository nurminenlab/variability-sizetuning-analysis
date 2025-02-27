import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params = pd.read_csv(F_dir + 'FA-params-May-2024.csv')

params = params[['layer',
                 'animal',
                 'fit_FA_SML',
                 'fit_FA_RF',
                 'fit_FA_SUR',
                 'fit_FA_SUR_200',
                 'fit_FA_SUR_400',
                 'fit_FA_SUR_800',
                 'fit_FA_LAR',
                 'fit_FA_MIN',
                 'fit_FA_MAX',
                 'fit_FA_BSL',
                 'fit_FA_MIN_diam',
                 'sur_MAX_diam',
                 'fit_RF']]

params = params.dropna()

params['RFnormed_maxFacilDiam']  = params['sur_MAX_diam'] / params['fit_RF']
params['RFnormed_maxQuenchDiam'] = params['fit_FA_MIN_diam'] / params['fit_RF']

FF_size = pd.DataFrame(columns=['fano','size','layer'])
FF_size_all = pd.DataFrame(columns=['fano','size','layer'])

var_dict =['0.1/0.2 deg','RF','2RF','4RF','8RF','26 deg']

params[var_dict[0]] = params['fit_FA_SML'] / params['fit_FA_RF']
params[var_dict[1]] = params['fit_FA_RF'] / params['fit_FA_RF']
params[var_dict[2]] = params['fit_FA_SUR_200'] / params['fit_FA_RF']
params[var_dict[3]] = params['fit_FA_SUR_400'] / params['fit_FA_RF']
params[var_dict[4]] = params['fit_FA_SUR_800'] / params['fit_FA_RF']
params[var_dict[5]] = params['fit_FA_LAR'] / params['fit_FA_RF']

FF_SML = pd.DataFrame(data={'fano':params[var_dict[0]].values,'size':['0.1/0.2 deg']*len(params.index),'layer':params['layer'].values})
FF_RF = pd.DataFrame(data={'fano':params[var_dict[1]].values,'size':['RF']*len(params.index),'layer':params['layer'].values})
FF_near_SUR_200 = pd.DataFrame(data={'fano':params[var_dict[2]].values,'size':['2RF']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_400 = pd.DataFrame(data={'fano':params[var_dict[3]].values,'size':['4RF']*len(params.index),'layer':params['layer'].values})
FF_far_SUR_800 = pd.DataFrame(data={'fano':params[var_dict[4]].values,'size':['8RF']*len(params.index),'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'fano':params[var_dict[5]].values,'size':['26 deg']*len(params.index),'layer':params['layer'].values})

FF_size = FF_size.append(FF_SML)
FF_size = FF_size.append(FF_RF)
FF_size = FF_size.append(FF_near_SUR_200)
FF_size = FF_size.append(FF_far_SUR_400)
FF_size = FF_size.append(FF_far_SUR_800)
FF_size = FF_size.append(FF_LAR)

funk_switch = 0
functions_list = [np.mean,np.median]
#functions_list = [sts.mstats.gmean,np.median]
agg_dict_gmean = {key: functions_list[funk_switch] for key in var_dict}

plt.figure()

ax = plt.subplot(111)
# this is a placeholder for the bootstrapped SEM
SEM = params.groupby('layer')[var_dict].sem()

layer_list = ['SG','G','IG']
p = params.groupby('layer')
# layer loop 
for l in layer_list:
    for cond in SEM.columns:
        if cond == var_dict[1]:
            SEM[cond][l] = 0 # no error for RF because it's normalized to itself
        else:
            SEM[cond][l] = sts.bootstrap((p[cond].get_group(l).values,),functions_list[funk_switch]).standard_error

params.groupby('layer').agg(agg_dict_gmean).plot(ax=ax,kind='bar',yerr=SEM)
ax.set_ylabel('Normalized fano-factor')    
if save_figures:
    plt.savefig(fig_dir + 'F5B-top.svg',bbox_inches='tight',pad_inches=0)

plt.figure()

ax = plt.subplot(111)
sns.swarmplot(x='layer',y='fano',hue='size',data=FF_size,ax=ax,size=3,dodge=True)
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'gray'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'gray'},
    'capprops':{'color':'gray'}}
sns.boxplot(x='layer',y='fano',hue='size',data=FF_size,ax=ax,showfliers=False,**PROPS)
ax.set_ylabel('Normalized fano-factor')
ax.legend('')
if save_figures:
    plt.savefig(fig_dir + 'F5B-bottom.svg',bbox_inches='tight',pad_inches=0)


FFsuppression = -100 *(1-(params['fit_FA_RF'] / params['fit_FA_BSL']))
FFsurfac = 100 * (params['fit_FA_LAR'] - params['fit_FA_RF'])/ params['fit_FA_RF']
FFsurfac_SUR = 100 * (params['fit_FA_SUR'] - params['fit_FA_RF'])/ params['fit_FA_RF']
FFsurfac_SUR_200 = 100 * (params['fit_FA_SUR_200'] - params['fit_FA_RF'])/ params['fit_FA_RF']
params.insert(3,'FFsuppression',FFsuppression.values)
params.insert(3,'FFsurfac',FFsurfac.values)
params.insert(3,'FFsurfac_SUR',FFsurfac_SUR.values)
params.insert(3,'FFsurfac_SUR_200',FFsurfac_SUR_200.values)

G  = params.query('layer == "G"')
IG = params.query('layer == "IG"')
SG = params.query('layer == "SG"')

plt.figure()
# BSL vs RF
#-------------------------------------------
# bootstrap errors

SEM_ffsupr = params.groupby('layer')['FFsuppression'].sem() # create appropriate table to store bootstrapped errors
SEM_ffsupr['G'] = sts.bootstrap((G['FFsuppression'].values,),np.median).standard_error
SEM_ffsupr['IG'] = sts.bootstrap((IG['FFsuppression'].values,),np.median).standard_error
SEM_ffsupr['SG'] = sts.bootstrap((SG['FFsuppression'].values,),np.median).standard_error

inds = params[params['FFsuppression'] > 300].index
params.drop(inds,inplace=True)
ax = plt.subplot(121)
params.groupby('layer')['FFsuppression'].median().plot(kind='bar',ax=ax,yerr=SEM_ffsupr,color='white',edgecolor='red')

ax.set_ylim(-100,200)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsuppression',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-100,200)
ax.set_title('BSL vs RF')
if save_figures:
    plt.savefig(fig_dir + 'F5C-left.svg',bbox_inches='tight',pad_inches=0)

print('\n t-test FFsuppression LSG vs L4C')
print(sts.ttest_ind(params[params['layer'] == 'L4C']['FFsuppression'],params[params['layer'] == 'LSG']['FFsuppression']))
print('\n t-test FFsuppression LSG vs LIG')
print(sts.ttest_ind(params[params['layer'] == 'LIG']['FFsuppression'],params[params['layer'] == 'LSG']['FFsuppression']))

print('\n ANOVA: the effect of layer on FFsuppression')
lm = ols('FFsuppression ~ C(layer)',data=params).fit()
print(sm.stats.anova_lm(lm,typ=1))

print(params['FFsurfac'].min())
inds = params[params['FFsurfac'] > 300].index
params.drop(inds,inplace=True)

# point stats and tests for each layer
print('##############################################')
print('BSL vs RF\n')
print('medians')
print(params.groupby('layer')['FFsuppression'].median())
print('\nbootstrapped errors for the medians')
print(SEM_ffsupr)

# create zero median distributions for bootstrap
SG_zeromedian = SG['FFsuppression'] - SG['FFsuppression'].median()
G_zeromedian  = G['FFsuppression'] - G['FFsuppression'].median()
IG_zeromedian = IG['FFsuppression'] - IG['FFsuppression'].median()

SG.insert(3,'FFsuppression_zeromedian',SG_zeromedian.values)
G.insert(3,'FFsuppression_zeromedian',G_zeromedian.values)
IG.insert(3,'FFsuppression_zeromedian',IG_zeromedian.values)

SG_boot = sts.bootstrap((SG['FFsuppression_zeromedian'].values,),np.median).bootstrap_distribution
G_boot  = sts.bootstrap((G['FFsuppression_zeromedian'].values,),np.median).bootstrap_distribution
IG_boot = sts.bootstrap((IG['FFsuppression_zeromedian'].values,),np.median).bootstrap_distribution


print('\np-values')
print('SG p BSL-RF: ', np.sum(SG_boot < SG['FFsuppression'].median())/len(SG_boot))
print('G p BSL-RF: ', np.sum(G_boot < G['FFsuppression'].median())/len(G_boot))
print('IG p BSL-RF: ', np.sum(IG_boot < IG['FFsuppression'].median())/len(IG_boot))

print('\n##############################################\n')

# 2*RF
#-------------------------------------------
plt.figure()

SEM_ffsurfac_SUR_200 = params.groupby('layer')['FFsurfac_SUR_200'].sem() # create appropriate table to store bootstrapped errors
""" SEM_ffsurfac_SUR_200['G'] = sts.bootstrap((G['FFsurfac_SUR_200'].values,),np.mean).standard_error
SEM_ffsurfac_SUR_200['IG'] = sts.bootstrap((IG['FFsurfac_SUR_200'].values,),np.mean).standard_error
SEM_ffsurfac_SUR_200['SG'] = sts.bootstrap((SG['FFsurfac_SUR_200'].values,),np.median).standard_error """

ax = plt.subplot(121)
params.groupby('layer')['FFsurfac_SUR_200'].mean().plot(kind='bar',ax=ax,yerr=SEM_ffsurfac_SUR_200,color='white',edgecolor='red')
ax.set_ylim(-100,300)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac_SUR_200',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-100,300)
ax.set_title('RF vs 2 * RF')
if save_figures:
    plt.savefig(fig_dir + 'F5D-NEAR_SUR.svg',bbox_inches='tight',pad_inches=0)


# point stats and tests for each layer
print('RF vs 2*RF\n')
print('medians')
print(params.groupby('layer')['FFsurfac_SUR_200'].median())
print('\nbootstrapped errors for the medians')
print(SEM_ffsurfac_SUR_200)

# create zero median distributions for bootstrap
SG_zeromedian = SG['FFsurfac_SUR_200'] - SG['FFsurfac_SUR_200'].median()
G_zeromedian  = G['FFsurfac_SUR_200'] - G['FFsurfac_SUR_200'].median()
IG_zeromedian = IG['FFsurfac_SUR_200'] - IG['FFsurfac_SUR_200'].median()

SG.insert(3,'FFsurfac_SUR_200_zeromedian',SG_zeromedian.values)
G.insert(3,'FFsurfac_SUR_200_zeromedian',G_zeromedian.values)
IG.insert(3,'FFsurfac_SUR_200_zeromedian',IG_zeromedian.values)

SG_boot = sts.bootstrap((SG['FFsurfac_SUR_200_zeromedian'].values,),np.median).bootstrap_distribution
G_boot  = sts.bootstrap((G['FFsurfac_SUR_200_zeromedian'].values,),np.median).bootstrap_distribution
IG_boot = sts.bootstrap((IG['FFsurfac_SUR_200_zeromedian'].values,),np.median).bootstrap_distribution

print('\np-values')
print('SG p RF vs 2RF: ', np.sum(SG_boot < SG['FFsurfac_SUR_200'].median())/len(SG_boot))
print('G p RF vs 2RF: ', np.sum(G_boot < G['FFsurfac_SUR_200'].median())/len(G_boot))
print('IG p RF vs 2RF: ', np.sum(IG_boot < IG['FFsurfac_SUR_200'].median())/len(IG_boot))

print('\n##############################################\n')

# RF-surround
#-------------------------------------------
plt.figure()

SEM_ffsurfac_SUR = params.groupby('layer')['FFsurfac_SUR'].sem() # create appropriate table to store bootstrapped errors
""" SEM_ffsurfac_SUR['G'] = sts.bootstrap((G['FFsurfac_SUR'].values,),np.mean).standard_error
SEM_ffsurfac_SUR['IG'] = sts.bootstrap((IG['FFsurfac_SUR'].values,),np.mean).standard_error
SEM_ffsurfac_SUR['SG'] = sts.bootstrap((SG['FFsurfac_SUR'].values,),np.mean).standard_error """

ax = plt.subplot(121)
params.groupby('layer')['FFsurfac_SUR'].mean().plot(kind='bar',ax=ax,yerr=SEM_ffsurfac_SUR,color='white',edgecolor='red')
ax.set_ylim(-100,300)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac_SUR',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-100,300)
ax.set_title('RF vs Surround')
if save_figures:
    plt.savefig(fig_dir + 'F5C-SUR.svg',bbox_inches='tight',pad_inches=0)

# point stats and tests for each layer
print('RF vs RF-surround\n')
print('medians')
print(params.groupby('layer')['FFsurfac_SUR'].median())
print('\nbootstrapped errors for the medians')
print(SEM_ffsurfac_SUR)

# create zero median distributions for bootstrap
SG_zeromedian = SG['FFsurfac_SUR'] - SG['FFsurfac_SUR'].median()
G_zeromedian  = G['FFsurfac_SUR'] - G['FFsurfac_SUR'].median()
IG_zeromedian = IG['FFsurfac_SUR'] - IG['FFsurfac_SUR'].median()

SG.insert(3,'FFsurfac_SUR_zeromedian',SG_zeromedian.values)
G.insert(3,'FFsurfac_SUR_zeromedian',G_zeromedian.values)
IG.insert(3,'FFsurfac_SUR_zeromedian',IG_zeromedian.values)

SG_boot = sts.bootstrap((SG['FFsurfac_SUR_zeromedian'].values,),np.median).bootstrap_distribution
G_boot  = sts.bootstrap((G['FFsurfac_SUR_zeromedian'].values,),np.median).bootstrap_distribution
IG_boot = sts.bootstrap((IG['FFsurfac_SUR_zeromedian'].values,),np.median).bootstrap_distribution

print('\np-values')
print('SG p RF vs RF-surround: ', np.sum(SG_boot > SG['FFsurfac_SUR'].median())/len(SG_boot))
print('G p RF vs RF-surround: ', np.sum(G_boot > G['FFsurfac_SUR'].median())/len(G_boot))
print('IG p RF vs RF-surround: ', np.sum(IG_boot < IG['FFsurfac_SUR'].median())/len(IG_boot))

print('\n##############################################\n')


# 26
#-------------------------------------------
plt.figure()

SEM_ffsurfac = params.groupby('layer')['FFsurfac'].sem() # create appropriate table to store bootstrapped errors
""" SEM_ffsurfac['G'] = sts.bootstrap((G['FFsurfac'].values,),np.mean).standard_error
SEM_ffsurfac['IG'] = sts.bootstrap((IG['FFsurfac'].values,),np.mean).standard_error
SEM_ffsurfac['SG'] = sts.bootstrap((SG['FFsurfac'].values,),np.mean).standard_error """

ax = plt.subplot(121)
params.groupby('layer')['FFsurfac'].mean().plot(kind='bar',ax=ax,yerr=SEM_ffsurfac,color='white',edgecolor='red')
ax.set_ylim(-100,300)
ax = plt.subplot(122)
sns.swarmplot(x='layer',y='FFsurfac',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_ylim(-100,300)
ax.set_title('RF vs 26')
if save_figures:
    plt.savefig(fig_dir + 'F5C-LAR.svg',bbox_inches='tight',pad_inches=0)


# point stats and tests for each layer
print('RF vs 26\n')
print('medians')
print(params.groupby('layer')['FFsurfac'].median())
print('\nbootstrapped errors for the medians')
print(SEM_ffsurfac)

# create zero median distributions for bootstrap
SG_zeromedian = SG['FFsurfac'] - SG['FFsurfac'].median()
G_zeromedian  = G['FFsurfac'] - G['FFsurfac'].median()
IG_zeromedian = IG['FFsurfac'] - IG['FFsurfac'].median()

SG.insert(3,'FFsurfac_zeromedian',SG_zeromedian.values)
G.insert(3,'FFsurfac_zeromedian',G_zeromedian.values)
IG.insert(3,'FFsurfac_zeromedian',IG_zeromedian.values)

SG_boot = sts.bootstrap((SG['FFsurfac_zeromedian'].values,),np.median).bootstrap_distribution
G_boot  = sts.bootstrap((G['FFsurfac_zeromedian'].values,),np.median).bootstrap_distribution
IG_boot = sts.bootstrap((IG['FFsurfac_zeromedian'].values,),np.median).bootstrap_distribution

print('\np-values')
print('SG p RF vs 26: ', np.sum(SG_boot > SG['FFsurfac'].median())/len(SG_boot))
print('G p RF vs 26: ', np.sum(G_boot > G['FFsurfac'].median())/len(G_boot))
print('IG p RF vs 26: ', np.sum(IG_boot < IG['FFsurfac'].median())/len(IG_boot))

print('\n##############################################\n')





# RFnormed_maxFacilDiam
#-------------------------------------------
SG_df = params.query('layer == "SG"')
G_df  = params.query('layer == "G"')
IG_df = params.query('layer == "IG"')

G_boot  = sts.bootstrap((G_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.99)
IG_boot = sts.bootstrap((IG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.99)
SG_boot = sts.bootstrap((SG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.99)

G_median  = G_boot.confidence_interval
IG_median = IG_boot.confidence_interval
SG_median = SG_boot.confidence_interval

medians = np.nan * np.ones((2,3))
medians[0,:] = np.array([G_median.low,IG_median.low,SG_median.low])
medians[1,:] = np.array([G_median.high,IG_median.high,SG_median.high])

for i in range(medians.shape[0]):
    medians[i,:] = np.abs(medians[i,:]- np.array([G_df['RFnormed_maxFacilDiam'].median(),
                                                  IG_df['RFnormed_maxFacilDiam'].median(),
                                                  SG_df['RFnormed_maxFacilDiam'].median()]))

plt.figure()
ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxFacilDiam'].median().plot(kind='bar',yerr=medians,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxFacilDiam',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F5Gii.svg')

Gp = np.sum(G_boot.bootstrap_distribution - G_df['RFnormed_maxFacilDiam'].median() + 1 > G_df['RFnormed_maxFacilDiam'].median()) / len(G_boot.bootstrap_distribution)
IGp = np.sum(IG_boot.bootstrap_distribution - IG_df['RFnormed_maxFacilDiam'].median() + 1 > IG_df['RFnormed_maxFacilDiam'].median()) / len(IG_boot.bootstrap_distribution)
SGp = np.sum(SG_boot.bootstrap_distribution - SG_df['RFnormed_maxFacilDiam'].median() + 1 > SG_df['RFnormed_maxFacilDiam'].median()) / len(SG_boot.bootstrap_distribution)

print('RFnormed_maxFacilDiam')
print('SG: ',SG_df['RFnormed_maxFacilDiam'].median())
print('G: ',G_df['RFnormed_maxFacilDiam'].median())
print('IG: ',IG_df['RFnormed_maxFacilDiam'].median())

print('RFnormed_maxFacilDiam errors')
print('SG: ',SG_median.low, SG_median.high)
print('G: ',G_median.low, G_median.high)
print('IG: ',IG_median.low, IG_median.high)

print('p values for RFnormed_maxFacilDiam')
print('p of finding this median for SG: ', SGp)
print('p of finding this median for G: ', Gp)
print('p of finding this median for IG: ', IGp)

# RFnormed_maxQuenchDiam
#-------------------------------------------
SG_df = params.query('layer == "SG"')
G_df  = params.query('layer == "G"')
IG_df = params.query('layer == "IG"')

G_boot  = sts.bootstrap((G_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian,confidence_level=0.99)
IG_boot = sts.bootstrap((IG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian,confidence_level=0.99)
SG_boot = sts.bootstrap((SG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian,confidence_level=0.99)

G_median = G_boot.confidence_interval
IG_median = IG_boot.confidence_interval
SG_median = SG_boot.confidence_interval


medians = np.nan * np.ones((2,3))
medians[0,:] = np.array([G_median.low,IG_median.low,SG_median.low])
medians[1,:] = np.array([G_median.high,IG_median.high,SG_median.high])

for i in range(medians.shape[0]):
    medians[i,:] = np.abs(medians[i,:]- np.array([G_df['RFnormed_maxQuenchDiam'].median(),
                                                  IG_df['RFnormed_maxQuenchDiam'].median(),
                                                  SG_df['RFnormed_maxQuenchDiam'].median()]))

plt.figure()
ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxQuenchDiam'].median().plot(kind='bar',yerr=medians,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxQuenchDiam',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F5G.svg')

Gp  = np.sum(G_boot.bootstrap_distribution - G_df['RFnormed_maxQuenchDiam'].median() + 1 > G_df['RFnormed_maxQuenchDiam'].median()) / len(G_boot.bootstrap_distribution)
IGp = np.sum(IG_boot.bootstrap_distribution - IG_df['RFnormed_maxQuenchDiam'].median() + 1 > IG_df['RFnormed_maxQuenchDiam'].median()) / len(IG_boot.bootstrap_distribution)
SGp = np.sum(SG_boot.bootstrap_distribution - SG_df['RFnormed_maxQuenchDiam'].median() + 1 > SG_df['RFnormed_maxQuenchDiam'].median()) / len(SG_boot.bootstrap_distribution)

print('RFnormed_maxQuenchDiam')
print('SG: ',SG_df['RFnormed_maxQuenchDiam'].median())
print('G: ',G_df['RFnormed_maxQuenchDiam'].median())
print('IG: ',IG_df['RFnormed_maxQuenchDiam'].median())

print('RFnormed_maxQuenchDiam errors')
print('SG: ',SG_median.low, SG_median.high)
print('G: ',G_median.low, G_median.high)
print('IG: ',IG_median.low, IG_median.high)

print('p values for RFnormed_maxQuenchDiam')
print('p of finding this median for SG: ', SGp)
print('p of finding this median for G: ', Gp)
print('p of finding this median for IG: ', IGp)