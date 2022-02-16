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
anal_duration = 300
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


quencher_DF = pd.DataFrame(columns=['qtype_signi',
                                    'bsl',
                                    'bsl_FR',
                                    'layer',
                                    'maxquench_diam',
                                    'maxamplif_diam',
                                    'maxquench',
                                    'maxamplif',
                                    'maxquench_perc',
                                    'maxamplif_perc',
                                    'RFdiam'])

# loop SG units
indx  = 0
qindx = 0
cont  = 100.0
count_window = 100
nboots = 3000

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
    Resp       = np.nan * np.ones((mn_mtrx.shape[0]))
    for stim in range(mn_mtrx.shape[0]):
        Resp[stim] = np.mean(mn_mtrx[stim,first_tp:last_tp])
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end] / (eps + mn_mtrx[stim,bsl_begin:bsl_end]))
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
                                                                                                nboots=nboots)

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

    RFdiam = diams_all[np.argmax(Resp)]
    diams_all = diams_all[signi_all=='S']

    # 
    if np.min(delta_fano) < 0:
        maxquench_diam = diams_all[np.argmin(delta_fano)]
        maxquench      = np.min(delta_fano)
        maxquench_perc = np.min(delta_fano) / bsl_signi[np.argmin(delta_fano)]
    else: # no significant quenching
        maxquench_diam = np.nan
        maxquench      = np.nan
        maxquench_perc = np.nan
        
    if np.max(delta_fano) > 0:
        maxamplif_diam = diams_all[np.argmax(delta_fano)]
        maxamplif      = np.max(delta_fano)
        maxamplif_perc = np.max(delta_fano) / bsl_signi[np.argmax(delta_fano)]
    else: # no significant amplification
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
                'qtype_signi':qtype_signi,
                'maxquench_diam':maxquench_diam,
                'maxamplif_diam':maxamplif_diam,
                'maxquench':maxquench,
                'maxamplif':maxamplif,
                'maxquench_perc':maxquench_perc,
                'maxamplif_perc':maxamplif_perc,
                'RFdiam':RFdiam}
    
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
    Resp       = np.nan * np.ones((mn_mtrx.shape[0]))         
    for stim in range(mn_mtrx.shape[0]):
        Resp[stim] = np.mean(mn_mtrx[stim,first_tp:last_tp])
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end] / (eps + mn_mtrx[stim,bsl_begin:bsl_end]))
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
                                                                                                nboots=nboots)

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

    RFdiam = diams_all[np.argmax(Resp)]
    diams_all = diams_all[signi_all=='S']
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
                'qtype_signi':qtype_signi,
                'maxquench_diam':maxquench_diam,
                'maxamplif_diam':maxamplif_diam,
                'maxquench':maxquench,
                'maxamplif':maxamplif,
                'maxquench_perc':maxquench_perc,
                'maxamplif_perc':maxamplif_perc,
                'RFdiam':RFdiam}
    
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
    Resp       = np.nan * np.ones((mn_mtrx.shape[0]))
    for stim in range(mn_mtrx.shape[0]):
        Resp[stim] = np.mean(mn_mtrx[stim,first_tp:last_tp])
        fano = np.mean(vr_mtrx[stim,first_tp:last_tp][0:-1:count_window] / (eps + mn_mtrx[stim,first_tp:last_tp][0:-1:count_window]))
        bsl[stim]  = np.mean(vr_mtrx[stim,bsl_begin:bsl_end] / (eps + mn_mtrx[stim,bsl_begin:bsl_end]))
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

    RFdiam = diams_all[np.argmax(Resp)]
    diams_all = diams_all[signi_all=='S']
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
                'layer':'IG',                
                'qtype_signi':qtype_signi,
                'maxquench_diam':maxquench_diam,
                'maxamplif_diam':maxamplif_diam,
                'maxquench':maxquench,
                'maxamplif':maxamplif,
                'maxquench_perc':maxquench_perc,
                'maxamplif_perc':maxamplif_perc,
                'RFdiam':RFdiam}
    
    tmp_df = pd.DataFrame(para_tmp, index=[qindx])
    quencher_DF = quencher_DF.append(tmp_df,sort=True)
    qindx =+ 1


# save results
quencher_DF.to_csv('amplification_DF_division.csv')