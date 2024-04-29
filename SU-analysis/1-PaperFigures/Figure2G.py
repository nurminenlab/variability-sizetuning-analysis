import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'
params = pd.read_csv(F_dir + 'SU-extracted_params-Jul2023.csv')

params['RFnormed_maxQuenchDiam'] = params['fit_fano_MIN_diam'] / params['fit_RF']
params = params[params['layer'] != 'L4C']

SG_df = params.query('layer == "LSG"')
IG_df = params.query('layer == "LIG"')

SEM = params.groupby('layer')['RFnormed_maxQuenchDiam'].sem()
SEM['LSG'] = sts.bootstrap((SG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian).standard_error
SEM['LIG'] = sts.bootstrap((IG_df['RFnormed_maxQuenchDiam'].values,),np.nanmedian).standard_error

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxQuenchDiam'].median().plot(kind='bar',yerr=SEM,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxQuenchDiam',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2G.svg')


print('RF_normed_maxQuenchDiam medians')
print(params.groupby('layer')['RFnormed_maxQuenchDiam'].median())

print('RF_normed_maxQuenchDiam bootstrapper errors for medians')
print(SEM)




