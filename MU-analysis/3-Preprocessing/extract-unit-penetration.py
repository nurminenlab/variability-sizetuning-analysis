# extract and save penetration for each unit

import pandas as pd
import numpy as np

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'

params = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')

SG = params[params['layer'] == 'LSG']
G  = params[params['layer'] == 'L4C']
IG = params[params['layer'] == 'LIG']

SG_penets = np.zeros(len(SG))
G_penets  = np.zeros(len(G))
IG_penets = np.zeros(len(IG))

SG_penets[SG['anipe'] == 'MK366P1'] = 1
SG_penets[SG['anipe'] == 'MK366P3'] = 2
SG_penets[SG['anipe'] == 'MK366P8'] = 3
SG_penets[SG['anipe'] == 'MK374P1'] = 4
SG_penets[SG['anipe'] == 'MK374P2'] = 5

G_penets[G['anipe'] == 'MK366P1'] = 1
G_penets[G['anipe'] == 'MK366P3'] = 2
G_penets[G['anipe'] == 'MK366P8'] = 3
G_penets[G['anipe'] == 'MK374P1'] = 4
G_penets[G['anipe'] == 'MK374P2'] = 5

IG_penets[IG['anipe'] == 'MK366P1'] = 1
IG_penets[IG['anipe'] == 'MK366P3'] = 2
IG_penets[IG['anipe'] == 'MK366P8'] = 3
IG_penets[IG['anipe'] == 'MK374P1'] = 4
IG_penets[IG['anipe'] == 'MK374P2'] = 5

np.save(F_dir + 'SG-penetration.npy', SG_penets)
np.save(F_dir + 'G-penetration.npy', G_penets)
np.save(F_dir + 'IG-penetration.npy', IG_penets)





