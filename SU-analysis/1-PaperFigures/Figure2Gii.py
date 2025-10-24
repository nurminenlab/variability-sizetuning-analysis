import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as sts

save_figures = False


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

def bootstrap_proportion_test(group1_values, group2_values, threshold=1, n_bootstrap=10000, alpha=0.05):
    """
    Bootstrap test for difference in proportions above threshold between two groups
    
    Parameters:
    -----------
    group1_values : array-like
        Values for first group
    group2_values : array-like  
        Values for second group
    threshold : float
        Threshold value (default=1)
    n_bootstrap : int
        Number of bootstrap samples
    alpha : float
        Significance level (default=0.05)
    
    Returns:
    --------
    dict with test results
    """
    
    # Calculate observed proportions
    prop1_obs = np.sum(group1_values > threshold) / len(group1_values)
    prop2_obs = np.sum(group2_values > threshold) / len(group2_values)
    observed_diff = prop1_obs - prop2_obs
    
    # Bootstrap under null hypothesis (no difference)
    # Pool all data and sample from combined pool
    combined_data = np.concatenate([group1_values, group2_values])
    n1, n2 = len(group1_values), len(group2_values)
    
    bootstrap_diffs = []
    
    for i in range(n_bootstrap):
        # Resample from combined pool
        resampled = np.random.choice(combined_data, size=n1+n2, replace=True)
        
        # Split into two groups of original sizes
        boot_group1 = resampled[:n1]
        boot_group2 = resampled[n1:]
        
        # Calculate proportions for this bootstrap sample
        boot_prop1 = np.sum(boot_group1 > threshold) / n1
        boot_prop2 = np.sum(boot_group2 > threshold) / n2
        boot_diff = boot_prop1 - boot_prop2
        
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
        'observed_diff': observed_diff,
        'prop_group1': prop1_obs,
        'prop_group2': prop2_obs,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha,
        'n_bootstrap': n_bootstrap,
        'bootstrap_diffs': bootstrap_diffs
    }

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-preprocessed/'
fig_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/SU-figures/'
params = pd.read_csv(F_dir + 'SU-extracted_params-Jul2023.csv')

params = params[params['layer'] != 'L4C']
units_to_remove = [1,7,14,51,53,58,80,20,31,32,34,46,77,81]
params = params[~params['unit'].isin(units_to_remove)]

params['RFnormed_maxFacilDiam'] = params['sur_MAX_diam'] / params['fit_RF']

SG_df = params.query('layer == "LSG"')
IG_df = params.query('layer == "LIG"')

SEM = params.groupby('layer')['RFnormed_maxFacilDiam'].sem()
SEM = np.zeros((2,2))
LSG_err = sts.bootstrap((SG_df['RFnormed_maxFacilDiam'].values,),
                           np.nanmedian,confidence_level=0.68).confidence_interval
LIG_err = sts.bootstrap((IG_df['RFnormed_maxFacilDiam'].values,),
                           np.nanmedian,confidence_level=0.68).confidence_interval
SEM[0,0] = LIG_err[0]
SEM[1,0] = LIG_err[1]
SEM[0,1] = LSG_err[0]
SEM[1,1] = LSG_err[1]

ax_bar = plt.subplot(121)
params.groupby('layer')['RFnormed_maxFacilDiam'].median().plot(kind='bar',ax=ax_bar, color='white',edgecolor='red')
ax_bar.plot([0,0],[LIG_err[0],LIG_err[1]],color='black')
ax_bar.plot([1,1],[LSG_err[0],LSG_err[1]],color='black')
ax_bar.set_yscale('log')
ax_bar.set_ylim(0.01,100)


ax = plt.subplot(122)
sns.swarmplot(x='layer',y='RFnormed_maxFacilDiam',data=params,ax=ax,size=3,hue='anipe')
ax.set_yscale('log')
ax.set_ylim(0.01,100)

if save_figures:
    plt.savefig(fig_dir + 'F2Gii.svg')

""" {'observed_diff': 0.12605042016806722,
 'prop_group1': 0.7142857142857143,
 'prop_group2': 0.5882352941176471,
 'p_value': 0.3744,
 'ci_lower': -0.27521008403361347,
 'ci_upper': 0.29201680672268904,
 'significant': False,
 'n_bootstrap': 10000,
 'bootstrap_diffs': array([-0.14705882,  0.14915966,  0.05462185, ..., -0.08613445,
        -0.06302521, -0.13235294])} """

test_result = bootstrap_median_test(SG_df['RFnormed_maxFacilDiam'].values, 
                                  IG_df['RFnormed_maxFacilDiam'].values)
print(f"P-value: {test_result['p_value']:.4f}")

