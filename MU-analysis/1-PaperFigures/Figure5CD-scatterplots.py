import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params = pd.read_csv(F_dir + 'FA-params-Aug-2023.csv')

plt.figure()
ax = plt.subplot(111)
ax.plot([0,3.5],[0,3.5],'k-')
sns.scatterplot(x='fit_FA_BSL',y='fit_FA_RF',hue='layer',style='animal',data=params,ax=ax, s=12)
ax.set_xlim(0.01,3.5)
ax.set_ylim(0.01,3.5)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal')

if save_figures:
    plt.savefig(fig_dir + 'F5_FA-scatters-BSL-RF.svg',bbox_inches='tight',pad_inches=0)

plt.figure()
ax = plt.subplot(111)
ax.plot([0,1.75],[0,1.75],'k-')
sns.scatterplot(x='fit_FA_RF',y='fit_FA_LAR',hue='layer',style='animal',data=params,ax=ax, s=12)
ax.set_xlim(0.01,1.75)
ax.set_ylim(0.01,1.75)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_aspect('equal')

if save_figures:
    plt.savefig(fig_dir + 'F5_FA-scatters-RF-26.svg',bbox_inches='tight',pad_inches=0)
