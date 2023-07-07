import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'

params = pd.read_csv(F_dir + 'SU-extracted_params-Jul2023.csv')

# we clean up units without much fano factor tuning
SG_units_to_remove = [7,14,26,50,51,53,58,68,80]
IG_units_to_remove = [20,46,81]

# find row indices for units to remove
idx_to_remove = []
for unit in SG_units_to_remove:
    idx_to_remove.append(params.index[params['unit'] == unit][0])

for unit in IG_units_to_remove:
    idx_to_remove.append(params.index[params['unit'] == unit][0])


params.drop(idx_to_remove,axis=0,inplace=True)

plt.figure()
ax = plt.subplot(111)
ax.plot([0,7],[0,7],'k-')
sns.scatterplot(x='fit_fano_BSL',y='fit_fano_RF',hue='layer',data=params,ax=ax, s=12)
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.set_aspect('equal')

if save_figures:
    plt.savefig(fig_dir + 'F2C_fano-scatters-BSL-RF.svg',bbox_inches='tight',pad_inches=0)

plt.figure()
ax = plt.subplot(111)
ax.plot([0,7],[0,7],'k-')
sns.scatterplot(x='fit_fano_RF',y='fit_fano_LAR',hue='layer',data=params,ax=ax, s=12)
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.set_aspect('equal')

if save_figures:
    plt.savefig(fig_dir + 'F2C_fano-scatters-RF-26.svg',bbox_inches='tight',pad_inches=0)
