import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'
params = pd.read_csv(F_dir + 'SU-extracted_params-Jun2023.csv')

params['RFnormed_maxQuenchDiam'] = params['fit_fano_MIN_diam'] / params['RFdiam']

SG_df = params.query('layer == "LSG"')
G_df  = params.query('layer == "L4C"')
IG_df = params.query('layer == "LIG"')

SG_medians = np.nanmedian(np.random.choice(SG_df['RFnormed_maxQuenchDiam'].values,
                            size=(10000,SG_df['RFnormed_maxQuenchDiam'].values.shape[0]),replace=True),axis=1)

G_medians = np.nanmedian(np.random.choice(G_df['RFnormed_maxQuenchDiam'].values,
                            size=(10000,G_df['RFnormed_maxQuenchDiam'].values.shape[0]),replace=True),axis=1)

IG_medians = np.nanmedian(np.random.choice(IG_df['RFnormed_maxQuenchDiam'].values,
                            size=(10000,IG_df['RFnormed_maxQuenchDiam'].values.shape[0]),replace=True),axis=1)

medians = [np.std(G_medians),np.std(IG_medians),np.std(SG_medians)]

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxQuenchDiam'].median().plot(kind='bar',yerr=medians,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxQuenchDiam',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2G.svg')


print('RF_normed_maxQuenchDiam medians')
params.groupby('layer')['RFnormed_maxQuenchDiam'].median()

print('RF_normed_maxQuenchDiam bootstrapper errors for medians')
print('SG: ', medians[2])
print('G: ',medians[0])
print('IG: ',medians[1])



