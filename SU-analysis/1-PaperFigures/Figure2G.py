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
params = pd.read_csv(F_dir + 'SU-extracted_params-Jul2023.csv')

params['RFnormed_maxQuenchDiam'] = params['fit_fano_MIN_diam'] / params['fit_RF']
params = params[params['layer'] != 'L4C']
units_to_remove = [1,7,14,51,53,58,80,20,31,32,34,46,77,81]
params = params[~params['unit'].isin(units_to_remove)]

def bootstrap_median_test(group1_values, group2_values, n_bootstrap=10000, alpha=0.05):
    """
    Bootstrap test for difference in medians between two groups
    
    Parameters:
    -----------
    group1_values : array-like
        Values for first group
    group2_values : array-like  
        Values for second group
    n_bootstrap : int
        Number of bootstrap samples (default=10000)
    alpha : float
        Significance level (default=0.05)
    
    Returns:
    --------
    dict with test results
    """
    
    # Calculate observed difference in medians
    median1_obs = np.nanmedian(group1_values)
    median2_obs = np.nanmedian(group2_values)
    observed_diff = median1_obs - median2_obs
    
    # Bootstrap under null hypothesis (no difference)
    # Pool all data and sample from combined pool
    combined_data = np.concatenate([group1_values, group2_values])
    combined_data = combined_data[~np.isnan(combined_data)]  # Remove NaNs
    
    n1, n2 = len(group1_values), len(group2_values)
    bootstrap_diffs = []
    
    for i in range(n_bootstrap):
        # Resample from combined pool
        boot_group1 = np.random.choice(combined_data, size=n1, replace=True)
        boot_group2 = np.random.choice(combined_data, size=n2, replace=True)
        
        # Calculate medians for this bootstrap sample
        boot_median1 = np.nanmedian(boot_group1)
        boot_median2 = np.nanmedian(boot_group2)
        boot_diff = boot_median1 - boot_median2
        
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate p-value (two-tailed)
    p_value = 2 * min(
        np.sum(bootstrap_diffs >= abs(observed_diff)) / n_bootstrap,
        np.sum(bootstrap_diffs <= -abs(observed_diff)) / n_bootstrap
    )
    
    # Calculate confidence interval for the difference
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha/2))
    
    return {
        'median_group1': median1_obs,
        'median_group2': median2_obs,
        'observed_diff': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha
    }

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
sns.swarmplot(x='layer',y='RFnormed_maxQuenchDiam',hue='anipe',data=params,ax=ax,size=3)
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2G.svg')


print('RF_normed_maxQuenchDiam medians')
print(params.groupby('layer')['RFnormed_maxQuenchDiam'].median())

print('RF_normed_maxQuenchDiam bootstrapper errors for medians')
print(SEM)

test_result = bootstrap_median_test(SG_df['RFnormed_maxQuenchDiam'].values, 
                                  IG_df['RFnormed_maxQuenchDiam'].values)
print(f"P-value: {test_result['p_value']:.4f}")



