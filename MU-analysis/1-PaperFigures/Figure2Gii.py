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
params = pd.read_csv(F_dir + 'SU-extracted_params-nosuppression-Apr2024.csv')

params['RFnormed_maxFacilDiam'] = params['sur_MAX_diam'] / params['fit_RF']
#params = params[params['layer'] != 'L4C']

SG_df = params.query('layer == "LSG"')
#G_df  = params.query('layer == "L4C"')
IG_df = params.query('layer == "LIG"')

SEM = params.groupby('layer')['RFnormed_maxFacilDiam'].sem()
SEM['LSG']  = sts.bootstrap((SG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian).standard_error
SEM['LIG']  = sts.bootstrap((IG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian).standard_error

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxFacilDiam'].median().plot(kind='bar',yerr=SEM,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxFacilDiam',hue='animal',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2Gii.svg')

print('RF_normed_maxFacilDiam medians')
print(params.groupby('layer')['RFnormed_maxFacilDiam'].median())

print('RF_normed_maxQuenchDiam bootstrapper errors for medians')
print(SEM)

