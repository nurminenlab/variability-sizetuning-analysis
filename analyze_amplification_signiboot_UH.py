import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import seaborn as sns
sys.path.append('C:/Users/lonurmin/Desktop/code/DataAnalysis/')
import data_analysislib as dalib
import statsmodels.api as sm
from statsmodels.formula.api import ols

import pdb

S_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/'
MUdatfile = 'selectedData_MUA_lenient_400ms_macaque_July-2020.pkl'

# analysis done between these timepoints
anal_duration = 400
first_tp  = 450
last_tp   = first_tp + anal_duration
bsl_begin = 120
bsl_end   = bsl_begin + anal_duration

eps = 0.0000001

with open(F_dir + MUdatfile,'rb') as f:
    data = pkl.load(f)

with open(S_dir + 'mean_PSTHs_SG-MK-MU.pkl','rb') as f:
    diams_data = pkl.load(f)

diams = np.array(list(diams_data.keys()))
del(diams_data)

with open(S_dir + 'mean_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_SG-MK-MU-Dec-2021.pkl','rb') as f:
    SG_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_mn_data = pkl.load(f)
with open(S_dir + 'vari_PSTHs_G-MK-MU-Dec-2021.pkl','rb') as f:
    G_vr_data = pkl.load(f)
    
with open(S_dir + 'mean_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_mn_data = pkl.load(f)    
with open(S_dir + 'vari_PSTHs_IG-MK-MU-Dec-2021.pkl','rb') as f:
    IG_vr_data = pkl.load(f)    

# param tables
SG_params = pd.DataFrame(columns=['fano',
                                  'bsl',
                                  'delta_fano',
                                  'diam',
                                  'unit',
                                  'bsl_FR',
                                  'layer',
                                  'signi'])

# param tables
G_params = pd.DataFrame(columns=['fano',
                                 'bsl',
                                 'delta_fano',
                                 'diam',
                                 'unit',
                                 'bsl_FR',
                                 'layer',
                                 'signi'])

# param tables
IG_params = pd.DataFrame(columns=['fano',
                                  'bsl',
                                  'delta_fano',
                                  'diam',
                                  'unit',
                                  'bsl_FR',
                                  'layer',
                                  'signi'])


quencher_DF = pd.DataFrame(columns=['qtype',
                                    'qtype_signi',
                                    'bsl',
                                    'bsl_FR',
                                    'layer',
                                    'maxquench_diam',
                                    'maxamplif_diam',
                                    'maxquench',
                                    'maxamplif',
                                    'maxquench_perc',
                                    'maxamplif_perc'])

# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100

SG_perc_amplif = np.zeros((len(list(SG_mn_data.keys())),19))
SG_perc_quench = np.zeros((len(list(SG_mn_data.keys())),19))

G_perc_amplif = np.zeros((len(list(G_mn_data.keys())),19))
G_perc_quench = np.zeros((len(list(G_mn_data.keys())),19))

IG_perc_amplif = np.zeros((len(list(IG_mn_data.keys())),19))
IG_perc_quench = np.zeros((len(list(IG_mn_data.keys())),19))

for unit_indx, unit in enumerate(list(SG_mn_data.keys())):
    # loop diams
    mn_mtrx = SG_mn_data[unit]
    vr_mtrx = SG_vr_data[unit]

    delta_fano = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl        = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl_FR     = np.nan * np.ones((mn_mtrx.shape[0]))
    signi_all  = np.nan * np.ones((mn_mtrx.shape[0]),dtype=object)
    for stim in range(mn_mtrx.shape[0]):
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp],axis=0) / (eps + np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end],axis=0) / (eps + np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0))
        bsl_FR[stim] = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
        delta_fano[stim] = fano - bsl[stim]
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]
            

        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=3000)

        fano_boot = np.mean(vari_PSTH_booted[:,first_tp:last_tp],axis=1) / (eps + np.mean(mean_PSTH_booted[:,first_tp:last_tp],axis=1))
        fano_boot = fano_boot - np.mean(fano_boot)
            
        if delta_fano[stim] > 0:
            if delta_fano[stim] >= np.percentile(fano_boot,95):
                signi = 'S'
            else:
                signi = 'NS'
        else:
            if delta_fano[stim] <= np.percentile(fano_boot,5):
                signi = 'S'
            else:
                signi = 'NS'

        signi_all[stim] = signi
        para_tmp  = {'fano':fano,'bsl':bsl[stim],'delta_fano':delta_fano[stim],'diam':diam,'unit':unit,'layer':'SG','signi':signi}
        tmp_df    = pd.DataFrame(para_tmp, index=[indx])
        SG_params = SG_params.append(tmp_df,sort=True)

        indx += 1


    if all(delta_fano < 0):
        qtype = 'quencher'
    elif all(delta_fano > 0):
        qtype = 'amplifier'
    else:
        qtype = 'mixer'

    if signi_all.shape[0] == 19:
        # amplification
        b_vec = np.logical_and(signi_all == 'S', delta_fano > 0)
        SG_perc_amplif[unit_indx,b_vec] = delta_fano[b_vec] / bsl[b_vec]
        # quenching
        b_vec = np.logical_and(signi_all == 'S', delta_fano < 0)
        SG_perc_quench[unit_indx,b_vec] = delta_fano[b_vec] / bsl[b_vec]
    else:
        # amplification
        b_vec = np.concatenate((np.reshape(False,(1,)),np.logical_and(signi_all == 'S', delta_fano > 0)))
        delta_fano_tmp = np.concatenate((np.reshape(np.nan,(1,)),delta_fano))
        bsl_tmp = np.concatenate((np.reshape(np.nan,(1,)),bsl))
        SG_perc_amplif[unit_indx,b_vec] = delta_fano_tmp[b_vec] / bsl_tmp[b_vec]
        # quenching
        b_vec = np.concatenate((np.reshape(False,(1,)),np.logical_and(signi_all == 'S', delta_fano < 0)))
        delta_fano_tmp = np.concatenate((np.reshape(np.nan,(1,)),delta_fano))
        bsl_tmp = np.concatenate((np.reshape(np.nan,(1,)),bsl))
        SG_perc_quench[unit_indx,b_vec] = delta_fano_tmp[b_vec] / bsl_tmp[b_vec]
        
        
    # use only significant data
    delta_fano = delta_fano[signi_all=='S']
    bsl_signi  = bsl[signi_all=='S']
    if mn_mtrx.shape[0] == 18:
        diams_all  = diams[1:]
    else:
        diams_all  = diams


    # 
    if np.min(delta_fano) < 0:
        maxquench_diam = diams_all[np.argmin(delta_fano)]
        maxquench      = np.min(delta_fano)
        maxquench_perc = np.min(delta_fano) / bsl_signi[np.argmin(delta_fano)]
    else:
        maxquench_diam = np.nan
        maxquench      = np.nan
        maxquench_perc = np.nan
        
    if np.max(delta_fano) > 0:
        maxamplif_diam = diams_all[np.argmax(delta_fano)]
        maxamplif      = np.max(delta_fano)
        maxamplif_perc = np.max(delta_fano) / bsl_signi[np.argmax(delta_fano)]
    else:
        maxamplif_diam = np.nan
        maxamplif      = np.nan
        maxamplif_perc = np.nan
        
    if all(delta_fano < 0):
        qtype_signi = 'quencher'
    elif all(delta_fano > 0):
        qtype_signi = 'amplifier'
    else:
        qtype_signi = 'mixer'

    para_tmp = {'bsl':np.mean(bsl),
                'bsl_FR':np.mean(bsl_FR),
                'layer':'SG',
                'qtype':qtype,
                'qtype_signi':qtype_signi,
                'maxquench_diam':maxquench_diam,
                'maxamplif_diam':maxamplif_diam,
                'maxquench':maxquench,
                'maxamplif':maxamplif,
                'maxquench_perc':maxquench_perc,
                'maxamplif_perc':maxamplif_perc}
    
    tmp_df = pd.DataFrame(para_tmp, index=[qindx])
    quencher_DF = quencher_DF.append(tmp_df,sort=True)
    qindx =+ 1


# loop G units
for unit_indx, unit in enumerate(list(G_mn_data.keys())):
    # loop diams
    mn_mtrx = G_mn_data[unit]
    vr_mtrx = G_vr_data[unit]
    delta_fano = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl        = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl_FR     = np.nan * np.ones((mn_mtrx.shape[0]))
    signi_all  = np.nan * np.ones((mn_mtrx.shape[0]),dtype=object)
    for stim in range(mn_mtrx.shape[0]):
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp],axis=0) / (eps + np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end],axis=0) / (eps + np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0))
        bsl_FR[stim] = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
        delta_fano[stim] = fano - bsl[stim]
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]


        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=3000)

        fano_boot = np.mean(vari_PSTH_booted[:,first_tp:last_tp],axis=1) / (eps + np.mean(mean_PSTH_booted[:,first_tp:last_tp],axis=1))
        fano_boot = fano_boot - np.mean(fano_boot)
        if delta_fano[stim] > 0:
            if delta_fano[stim] >= np.percentile(fano_boot,95):
                signi = 'S'
            else:
                signi = 'NS'
        else:
            if delta_fano[stim] <= np.percentile(fano_boot,5):
                signi = 'S'
            else:
                signi = 'NS'

        signi_all[stim] = signi
        para_tmp = {'fano':fano,'bsl':bsl[stim],'delta_fano':delta_fano[stim],'diam':diam,'unit':unit,'layer':'G','signi':signi}
        tmp_df   = pd.DataFrame(para_tmp, index=[indx])
        G_params = G_params.append(tmp_df,sort=True)

        indx += 1


    if all(delta_fano < 0):
        qtype = 'quencher'
    elif all(delta_fano > 0):
        qtype = 'amplifier'
    else:
        qtype = 'mixer'


    if signi_all.shape[0] == 19:
        # amplification
        b_vec = np.logical_and(signi_all == 'S', delta_fano > 0)
        G_perc_amplif[unit_indx,b_vec] = delta_fano[b_vec] / bsl[b_vec]
        # quenching
        b_vec = np.logical_and(signi_all == 'S', delta_fano < 0)
        G_perc_quench[unit_indx,b_vec] = delta_fano[b_vec] / bsl[b_vec]
    else:
        # amplification
        b_vec = np.concatenate((np.reshape(False,(1,)),np.logical_and(signi_all == 'S', delta_fano > 0)))
        delta_fano_tmp = np.concatenate((np.reshape(np.nan,(1,)),delta_fano))
        bsl_tmp = np.concatenate((np.reshape(np.nan,(1,)),bsl))
        G_perc_amplif[unit_indx,b_vec] = delta_fano_tmp[b_vec] / bsl_tmp[b_vec]
        # quenching
        b_vec = np.concatenate((np.reshape(False,(1,)),np.logical_and(signi_all == 'S', delta_fano < 0)))
        delta_fano_tmp = np.concatenate((np.reshape(np.nan,(1,)),delta_fano))
        bsl_tmp = np.concatenate((np.reshape(np.nan,(1,)),bsl))
        G_perc_quench[unit_indx,b_vec] = delta_fano_tmp[b_vec] / bsl_tmp[b_vec]

        
    delta_fano = delta_fano[signi_all=='S']
    bsl_signi  = bsl[signi_all=='S']
    if mn_mtrx.shape[0] == 18:
        diams_all  = diams[1:]
    else:
        diams_all  = diams


    # 
    if np.min(delta_fano) < 0:
        maxquench_diam = diams_all[np.argmin(delta_fano)]
        maxquench      = np.min(delta_fano)
        maxquench_perc = np.min(delta_fano) / bsl_signi[np.argmin(delta_fano)]
    else:
        maxquench_diam = np.nan
        maxquench      = np.nan
        maxquench_perc = np.nan
        
    if np.max(delta_fano) > 0:
        maxamplif_diam = diams_all[np.argmax(delta_fano)]
        maxamplif      = np.max(delta_fano)
        maxamplif_perc = np.max(delta_fano) / bsl_signi[np.argmax(delta_fano)]
    else:
        maxamplif_diam = np.nan
        maxamplif      = np.nan
        maxamplif_perc = np.nan
        
    if all(delta_fano < 0):
        qtype_signi = 'quencher'
    elif all(delta_fano > 0):
        qtype_signi = 'amplifier'
    else:
        qtype_signi = 'mixer'


    para_tmp = {'bsl':np.mean(bsl),
                'bsl_FR':np.mean(bsl_FR),
                'layer':'G',
                'qtype':qtype,
                'qtype_signi':qtype_signi,
                'maxquench_diam':maxquench_diam,
                'maxamplif_diam':maxamplif_diam,
                'maxquench':maxquench,
                'maxamplif':maxamplif,
                'maxquench_perc':maxquench_perc,
                'maxamplif_perc':maxamplif_perc}
    
    tmp_df = pd.DataFrame(para_tmp, index=[qindx])
    quencher_DF = quencher_DF.append(tmp_df,sort=True)
    qindx =+ 1
        

# loop IG units
for unit_indx, unit in enumerate(list(IG_mn_data.keys())):
    # loop diams
    mn_mtrx = IG_mn_data[unit]
    vr_mtrx = IG_vr_data[unit]
    delta_fano = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl        = np.nan * np.ones((mn_mtrx.shape[0]))
    bsl_FR     = np.nan * np.ones((mn_mtrx.shape[0]))
    signi_all  = np.nan * np.ones((mn_mtrx.shape[0]),dtype=object)
    for stim in range(mn_mtrx.shape[0]):
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp],axis=0) / (eps + np.mean(mn_mtrx[stim,first_tp:last_tp],axis=0))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end],axis=0) / (eps + np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0))
        bsl_FR[stim] = np.mean(mn_mtrx[stim,bsl_begin:bsl_end],axis=0)
        delta_fano[stim] = fano - bsl[stim]
        
        if mn_mtrx.shape[0] == 18:
            diam = diams[stim+1]
        else:
            diam = diams[stim]


        mean_PSTH, vari_PSTH,binned_data,mean_PSTH_booted,vari_PSTH_booted = dalib.meanvar_PSTH(data[unit][cont]['spkR_NoL'][:,stim,:],
                                                                                                count_window,
                                                                                                style='same',
                                                                                                return_bootdstrs=True,
                                                                                                nboots=3000)

        fano_boot = np.mean(vari_PSTH_booted[:,first_tp:last_tp],axis=1) / (eps + np.mean(mean_PSTH_booted[:,first_tp:last_tp],axis=1))
        fano_boot = fano_boot - np.mean(fano_boot)
        if delta_fano[stim] > 0:
            if delta_fano[stim] >= np.percentile(fano_boot,95):
                signi = 'S'
            else:
                signi = 'NS'
        else:
            if delta_fano[stim] <= np.percentile(fano_boot,5):
                signi = 'S'
            else:
                signi = 'NS'

        signi_all[stim] = signi
        para_tmp = {'fano':fano,'bsl':bsl[stim],'delta_fano':delta_fano[stim],'diam':diam,'unit':unit,'layer':'IG','signi':signi}
        tmp_df   = pd.DataFrame(para_tmp, index=[indx])
        IG_params = IG_params.append(tmp_df,sort=True)

        indx += 1


    if all(delta_fano < 0):
        qtype = 'quencher'
    elif all(delta_fano > 0):
        qtype = 'amplifier'
    else:
        qtype = 'mixer'

    if signi_all.shape[0] == 19:
        # amplification
        b_vec = np.logical_and(signi_all == 'S', delta_fano > 0)
        IG_perc_amplif[unit_indx,b_vec] = delta_fano[b_vec] / bsl[b_vec]
        # quenching
        b_vec = np.logical_and(signi_all == 'S', delta_fano < 0)
        IG_perc_quench[unit_indx,b_vec] = delta_fano[b_vec] / bsl[b_vec]
    else:
        # amplification
        b_vec = np.concatenate((np.reshape(False,(1,)),np.logical_and(signi_all == 'S', delta_fano > 0)))
        delta_fano_tmp = np.concatenate((np.reshape(np.nan,(1,)),delta_fano))
        bsl_tmp = np.concatenate((np.reshape(np.nan,(1,)),bsl))
        IG_perc_amplif[unit_indx,b_vec] = delta_fano_tmp[b_vec] / bsl_tmp[b_vec]
        # quenching
        b_vec = np.concatenate((np.reshape(False,(1,)),np.logical_and(signi_all == 'S', delta_fano < 0)))
        delta_fano_tmp = np.concatenate((np.reshape(np.nan,(1,)),delta_fano))
        bsl_tmp = np.concatenate((np.reshape(np.nan,(1,)),bsl))
        IG_perc_quench[unit_indx,b_vec] = delta_fano_tmp[b_vec] / bsl_tmp[b_vec]
        
    # use significant data
    delta_fano = delta_fano[signi_all=='S']
    bsl_signi  = bsl[signi_all=='S']
    if mn_mtrx.shape[0] == 18:
        diams_all  = diams[1:]
    else:
        diams_all  = diams


    # 
    if np.min(delta_fano) < 0:
        maxquench_diam = diams_all[np.argmin(delta_fano)]
        maxquench      = np.min(delta_fano)
        maxquench_perc = np.min(delta_fano) / bsl_signi[np.argmin(delta_fano)]
    else:
        maxquench_diam = np.nan
        maxquench      = np.nan
        maxquench_perc = np.nan
        
    if np.max(delta_fano) > 0:
        maxamplif_diam = diams_all[np.argmax(delta_fano)]
        maxamplif      = np.max(delta_fano)
        maxamplif_perc = np.max(delta_fano) / bsl_signi[np.argmax(delta_fano)]
    else:
        maxamplif_diam = np.nan
        maxamplif      = np.nan
        maxamplif_perc = np.nan
        
        
    if all(delta_fano < 0):
        qtype_signi = 'quencher'
    elif all(delta_fano > 0):
        qtype_signi = 'amplifier'
    else:
        qtype_signi = 'mixer'


    if all(delta_fano < 0):
        qtype_signi = 'quencher'
    elif all(delta_fano > 0):
        qtype = 'amplifier'
    else:
        qtype = 'mixer'

    para_tmp = {'bsl':np.mean(bsl),
                'bsl_FR':np.mean(bsl_FR),
                'layer':'IG',
                'qtype':qtype,
                'qtype_signi':qtype_signi,
                'maxquench_diam':maxquench_diam,
                'maxamplif_diam':maxamplif_diam,
                'maxquench':maxquench,
                'maxamplif':maxamplif,
                'maxquench_perc':maxquench_perc,
                'maxamplif_perc':maxamplif_perc}
    
    tmp_df = pd.DataFrame(para_tmp, index=[qindx])
    quencher_DF = quencher_DF.append(tmp_df,sort=True)
    qindx =+ 1

my_palette = sns.color_palette(['gray','black'])        
plt.figure(1, figsize=(4.5, 1.5))
ax = plt.subplot(1,3,1)
sns.scatterplot(x='diam',y='delta_fano',data=SG_params,s=4,hue='signi',palette=my_palette,ax=ax)

ax.set_xscale('log')
ax.set_xticks([0.1, 1., 10])
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_ylim(-8,6)

ax = plt.subplot(1,3,2)
sns.scatterplot(x='diam',y='delta_fano',data=G_params,s=4,hue='signi',palette=my_palette,ax=ax)
ax.set_xscale('log')
ax.set_xticks([0.1, 1., 10])
ax.set_yticklabels([])
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_ylim(-8,6)

ax = plt.subplot(1,3,3)
my_palette = sns.color_palette(['black','gray'])
sns.scatterplot(x='diam',y='delta_fano',data=IG_params,s=4,hue='signi',palette=my_palette,ax=ax)
ax.set_xscale('log')
ax.set_xticks([0.1, 1., 10])
ax.set_yticklabels([])
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_ylim(-8,6)


plt.figure(2, figsize=(4.5, 1.5))
ax = plt.subplot(1,2,1)
quencher_DF.groupby(['layer','qtype_signi']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar', stacked=True, ax=ax,color=['red','grey'])

ax = plt.subplot(1,2,2)
SEM = quencher_DF.groupby(['layer','qtype_signi'])['bsl'].sem()
quencher_DF.groupby(['layer','qtype_signi'])['bsl'].mean().plot(kind='bar',yerr=SEM,ax=ax)

quencher_DF.to_csv('quencher_DF.csv')

lm = ols('bsl ~ C(qtype_signi) + C(layer)',data=quencher_DF).fit()
table = sm.stats.anova_lm(lm,typ=1)

SEM = quencher_DF.groupby('layer')['maxquench_diam','maxamplif_diam'].sem()
quencher_DF.groupby('layer')['maxquench_diam','maxamplif_diam'].mean().plot(kind='bar',yerr=SEM)

SEM = quencher_DF.groupby('layer')['maxquench_perc','maxamplif_perc'].sem()
quencher_DF.groupby('layer')['maxquench_perc','maxamplif_perc'].mean().plot(kind='bar',yerr=SEM)

quencher_DF['maxquench_diam'].hist(ec='black',fc='gray',bins=np.arange(0.1,4,0.2),grid=False)
quencher_DF['maxamplif_diam'].hist(ec='black',fc='gray',bins=np.arange(0.1,4,0.2),grid=False)

SG_perc_amplif = SG_perc_amplif * 100
SG_perc_quench = SG_perc_quench * 100

SG_perc_amplif_mn = np.mean(SG_perc_amplif,axis=0)
SG_perc_amplif_se = np.std(SG_perc_amplif,axis=0) / np.sqrt(SG_perc_amplif.shape[0])
SG_perc_amplif_ub = SG_perc_amplif_mn + SG_perc_amplif_se
SG_perc_amplif_lb = SG_perc_amplif_mn - SG_perc_amplif_se

SG_perc_quench_mn = np.mean(SG_perc_quench,axis=0)
SG_perc_quench_se = np.std(SG_perc_quench,axis=0) / np.sqrt(SG_perc_quench.shape[0])
SG_perc_quench_ub = SG_perc_quench_mn + SG_perc_quench_se
SG_perc_quench_lb = SG_perc_quench_mn - SG_perc_quench_se

G_perc_amplif = G_perc_amplif * 100
G_perc_quench = G_perc_quench * 100

G_perc_amplif_mn = np.mean(G_perc_amplif,axis=0)
G_perc_amplif_se = np.std(G_perc_amplif,axis=0) / np.sqrt(G_perc_amplif.shape[0])
G_perc_amplif_ub = G_perc_amplif_mn + G_perc_amplif_se
G_perc_amplif_lb = G_perc_amplif_mn - G_perc_amplif_se

G_perc_quench_mn = np.mean(G_perc_quench,axis=0)
G_perc_quench_se = np.std(G_perc_quench,axis=0) / np.sqrt(G_perc_quench.shape[0])
G_perc_quench_ub = G_perc_quench_mn + G_perc_quench_se
G_perc_quench_lb = G_perc_quench_mn - G_perc_quench_se

IG_perc_amplif = IG_perc_amplif * 100
IG_perc_quench = IG_perc_quench * 100

IG_perc_amplif_mn = np.mean(IG_perc_amplif,axis=0)
IG_perc_amplif_se = np.std(IG_perc_amplif,axis=0) / np.sqrt(IG_perc_amplif.shape[0])
IG_perc_amplif_ub = IG_perc_amplif_mn + IG_perc_amplif_se
IG_perc_amplif_lb = IG_perc_amplif_mn - IG_perc_amplif_se

IG_perc_quench_mn = np.mean(IG_perc_quench,axis=0)
IG_perc_quench_se = np.std(IG_perc_quench,axis=0) / np.sqrt(IG_perc_quench.shape[0])
IG_perc_quench_ub = IG_perc_quench_mn + IG_perc_quench_se
IG_perc_quench_lb = IG_perc_quench_mn - IG_perc_quench_se

# SG 
plt.figure(figsize=(4.5, 0.93))
plt.subplot(1,3,1)
plt.fill_between(diams,SG_perc_amplif_lb, SG_perc_amplif_ub,color='r')
plt.plot(diams,SG_perc_amplif_mn,'k--')
plt.fill_between(diams,SG_perc_quench_lb, SG_perc_quench_ub,color='b')
plt.plot(diams,SG_perc_quench_mn,'k--')
plt.ylim(-50, 50)
plt.xscale('log')
plt.xticks([0.1, 1, 10])

# G
plt.subplot(1,3,2)
plt.fill_between(diams,G_perc_amplif_lb, G_perc_amplif_ub,color='r')
plt.plot(diams,G_perc_amplif_mn,'k--')
plt.fill_between(diams,G_perc_quench_lb, G_perc_quench_ub,color='b')
plt.plot(diams,G_perc_quench_mn,'k--')
plt.ylim(-50, 50)
plt.xscale('log')
plt.xticks([0.1, 1, 10])

# IG
plt.subplot(1,3,3)
plt.fill_between(diams,IG_perc_amplif_lb, IG_perc_amplif_ub,color='r')
plt.plot(diams,IG_perc_amplif_mn,'k--')
plt.fill_between(diams,IG_perc_quench_lb, IG_perc_quench_ub,color='b')
plt.plot(diams,IG_perc_quench_mn,'k--')
plt.ylim(-50, 50)
plt.xscale('log')
plt.xticks([0.1, 1, 10])

# nboots = 10000
# boot_inds = np.random.choice(tmp.shape[0],(tmp.shape[0],nboots))

SG_maxquench_diam = quencher_DF[quencher_DF['layer'] == 'SG']['maxquench_diam'].values
SG_maxamplif_diam = quencher_DF[quencher_DF['layer'] == 'SG']['maxamplif_diam'].values

G_maxquench_diam = quencher_DF[quencher_DF['layer'] == 'G']['maxquench_diam'].values
G_maxamplif_diam = quencher_DF[quencher_DF['layer'] == 'G']['maxamplif_diam'].values

IG_maxquench_diam = quencher_DF[quencher_DF['layer'] == 'IG']['maxquench_diam'].values
IG_maxamplif_diam = quencher_DF[quencher_DF['layer'] == 'IG']['maxamplif_diam'].values

plt.figure(figsize=(1.481, 1.128))
ax = plt.subplot(1,1,1)
SEM = quencher_DF.groupby('layer')['maxquench','maxamplif'].sem()
quencher_DF.groupby('layer')['maxquench','maxamplif'].mean().plot(kind='bar',yerr=SEM,ax=ax)

plt.show()
