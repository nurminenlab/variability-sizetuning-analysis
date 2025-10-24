import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = True

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/MU-figures/'
params = pd.read_csv(F_dir + 'extracted_params-nearsurrounds-Jul2023.csv')

params['RFnormed_maxFacilDiam'] = params['sur_MAX_diam'] / params['fit_RF']
#params = params[params['layer'] != 'L4C']

SG_df = params.query('layer == "LSG"')
G_df  = params.query('layer == "L4C"')
IG_df = params.query('layer == "LIG"')

G_median = sts.bootstrap((G_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.99).confidence_interval
IG_median = sts.bootstrap((IG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.99).confidence_interval
SG_median = sts.bootstrap((SG_df['RFnormed_maxFacilDiam'].values,),np.nanmedian,confidence_level=0.99).confidence_interval

medians = np.nan * np.ones((2,3))
medians[0,:] = np.array([G_median.low,IG_median.low,SG_median.low])
medians[1,:] = np.array([G_median.high,IG_median.high,SG_median.high])

for i in range(medians.shape[0]):
    medians[i,:] = np.abs(medians[i,:]- np.array([G_df['RFnormed_maxFacilDiam'].median(),
                                                  IG_df['RFnormed_maxFacilDiam'].median(),
                                                  SG_df['RFnormed_maxFacilDiam'].median()]))

ax = plt.subplot(121)
params.groupby('layer')['RFnormed_maxFacilDiam'].median().plot(kind='bar',yerr=medians,ax=ax,color='white',edgecolor='red')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxFacilDiam',hue='anipe',data=params,ax=ax,size=3)
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2Gii.svg')

print('RF_normed_maxFacilDiam medians')
print(params.groupby('layer')['RFnormed_maxFacilDiam'].median())

print('RF_normed_maxQuenchDiam bootstrapped CI for medians')
print('Low: G, IG, SG ', [G_median.low,IG_median.low,SG_median.low])
print('High: G, IG, SG ', [G_median.high,IG_median.high,SG_median.high])

G_df['RFnormed_maxFacilDiam_median_one'] = G_df['RFnormed_maxFacilDiam'] - G_df['RFnormed_maxFacilDiam'].median() + 1
IG_df['RFnormed_maxFacilDiam_median_one'] = IG_df['RFnormed_maxFacilDiam'] - IG_df['RFnormed_maxFacilDiam'].median() + 1
SG_df['RFnormed_maxFacilDiam_median_one'] = SG_df['RFnormed_maxFacilDiam'] - SG_df['RFnormed_maxFacilDiam'].median() + 1

G_median  = sts.bootstrap((G_df['RFnormed_maxFacilDiam_median_one'].values,),np.nanmedian).bootstrap_distribution
IG_median = sts.bootstrap((IG_df['RFnormed_maxFacilDiam_median_one'].values,),np.nanmedian).bootstrap_distribution
SG_median = sts.bootstrap((SG_df['RFnormed_maxFacilDiam_median_one'].values,),np.nanmedian).bootstrap_distribution

print('p_value RF_normed_maxFacilDiam median, assuming true median is 1')
print('SG p-value: ', np.sum(SG_median > SG_df['RFnormed_maxFacilDiam'].median()) / len(SG_median))
print('G p-value: ', np.sum(G_median > G_df['RFnormed_maxFacilDiam'].median()) / len(G_median))
print('IG p-value: ', np.sum(IG_median > IG_df['RFnormed_maxFacilDiam'].median()) / len(IG_median))




