import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = False

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')

params['RFnormed_maxFacilDiam'] = params['sur_MAX_diam'] / params['fit_RF']

SG_df = params.query('layer == "LSG"')
G_df  = params.query('layer == "L4C"')
IG_df = params.query('layer == "LIG"')

SG_ci  = sts.bootstrap((SG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.68)
G_ci   = sts.bootstrap((G_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.68)
IG_ci  = sts.bootstrap((IG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.68)

medians = np.nan * np.ones((2,3))
# G
medians[0,0] = G_ci.confidence_interval[0]
medians[1,0] = G_ci.confidence_interval[1]
# IG 
medians[0,1] = IG_ci.confidence_interval[0]
medians[1,1] = IG_ci.confidence_interval[1]
# SG 
medians[0,2] = SG_ci.confidence_interval[0]
medians[1,2] = SG_ci.confidence_interval[1]

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxFacilDiam'].median().plot(kind='bar',yerr=medians,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxFacilDiam',data=params,ax=ax,size=3,color='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2G.svg')

