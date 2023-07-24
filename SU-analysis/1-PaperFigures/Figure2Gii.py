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

params = params[params['layer'] != 'L4C']

params['RFnormed_maxFacilDiam'] = params['sur_MAX_diam'] / params['fit_RF']

SG_df = params.query('layer == "LSG"')
IG_df = params.query('layer == "LIG"')

SEM = params.groupby('layer')['RFnormed_maxFacilDiam'].sem()
SEM['LSG'] = sts.bootstrap((SG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian).standard_error
SEM['LIG'] = sts.bootstrap((IG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian).standard_error

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxFacilDiam'].median().plot(kind='bar',ax=ax,yerr=SEM, color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)


ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxFacilDiam',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2Gii.svg')

