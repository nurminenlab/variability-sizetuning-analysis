import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'

params = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')

plt.figure()
ax = plt.subplot(111)
ax.plot([0,7],[0,7],'k-')
sns.scatterplot(x='fit_fano_BSL',y='fit_fano_RF',hue='layer',data=params,ax=ax, s=12, style='animal')
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.set_aspect('equal')

if save_figures:
    plt.savefig(fig_dir + 'F2F_fano-scatters-BSL-RF.svg',bbox_inches='tight',pad_inches=0)

plt.figure()
ax = plt.subplot(111)
ax.plot([0,7],[0,7],'k-')
sns.scatterplot(x='fit_fano_RF',y='fit_fano_LAR',hue='layer',data=params,ax=ax, s=12,style='animal')
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.set_aspect('equal')

if save_figures:
    plt.savefig(fig_dir + 'F2F_fano-scatters-RF-26.svg',bbox_inches='tight',pad_inches=0)

plt.figure()
ax = plt.subplot(111)
ax.plot([0,7],[0,7],'k-')
sns.scatterplot(x='fit_fano_RF',y='fit_fano_near_SUR_200',hue='layer',data=params,ax=ax, s=12,style='animal')
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.set_aspect('equal')

if save_figures:
    plt.savefig(fig_dir + 'F2F_fano-scatters-RF-near-SUR.svg',bbox_inches='tight',pad_inches=0)
