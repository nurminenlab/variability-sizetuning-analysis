import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns

F_dir   = 'C:/Users/lonurmin/Desktop/CorrelatedVariability/results/paper_v9/MK-MU/'
params = pd.read_csv(F_dir + 'extracted_params-Dec-2021.csv')

FF_size = pd.DataFrame(columns=['fano','size','layer'])

indx = params[params['fit_fano_MIN'] == 0].index
params.drop(indx,inplace=True)

FF_RF = pd.DataFrame(data={'fano':params['fit_fano_RF'].values,'size':['RF']*len(params.index),'layer':params['layer'].values})
FF_LAR = pd.DataFrame(data={'fano':params['fit_fano_LAR'].values,'size':['LAR']*len(params.index),'layer':params['layer'].values})

FF_size = FF_size.append(FF_RF)
FF_size = FF_size.append(FF_LAR)

plt.figure()
ax = plt.subplot(111)
SEM = params.groupby('layer')[['fit_fano_SML','fit_fano_RF','fit_fano_SUR','fit_fano_LAR']].sem()
params.groupby('layer')[['fit_fano_SML','fit_fano_RF','fit_fano_SUR','fit_fano_LAR']].mean().plot(ax=ax,kind='bar',yerr=SEM)

plt.figure()
ax = plt.subplot(111)
sns.swarmplot(x='layer',y='fano',hue='size',data=FF_size,ax=ax,size=3,dodge=True)
