
# Size tuning of neural response variability in laminar circuits of macaque primary visual cortex
---

This data set contains all the files needed to replicate the analyses in our paper 
https://doi.org/10.1101/2023.01.17.524397
The codes for the analyses presented in the paper can be found here
https://github.com/nurminenlab/variability-sizetuning-analysis


## Description of the data and file structure
The file, MUA_data.pkl, contains all the data that was published in the above paper. You will need python to open this file. 

'''
import pickle
F_dir   = '/path/to/the/datafile/'
datfile = 'MUA_data.pkl'
with open(F_dir + SUdatfile,'rb') as f:
    data = pkl.load(f)
'''

After running the commands above, the variable data contains spiking data of all 116 macaque V1 units that we recorded. 
len(data) will return the number of recorded units 
unit = 0
data[unit].keys() will return 
dict_keys([100.0, 'info']) the field info contains useful information call
data[unit]['info'].keys() for more info, this command will return
dict_keys(['diam', 'layer', 'animal', 'penetr'])
data[unit]['info']['diam'] returns the stimulus diameters used for recording the responses of this unit
data[unit]['info']['layer'] returns the cortical layer from which this unit was recorded
data[unit]['info']['animal'] returns the animal number
data[unit]['info']['penetration'] returns the penetration number

data[0][100.0].keys() returns (100.0 refers to the luminance contrast used in these recordings)
dict_keys(['spkC_NoL', 'baseline', 'fano_NoL', 'fano_bsl', 'boot_fano_NoL', 'boot_fano_bsl', 'fano_bins_NoL', 'spkR_NoL'])

spkR_NoL -> this numpy array contain everything you need, basically it has spike times binned in 1 ms bis, the bin takes value 1 if there was a spike and 0 otherwise
the dimensions are trial x stimulus diameter x time bin, 
for example calling data[0][100.0]['spkR_NoL'][0,0,0] would give you the first trial, smallest stimulus diameter (cross check with data[0]['info']['diam']), and the first time bin


the variables below are not directly used in the paper, but we have saved them, so use them if you wish to do so
the numpy arrays are organized in the same way as spkR_NoL
spkC_NoL -> spike count 0-500 ms from the stimulus onset
baseline -> baseline spikecount 
fano_NoL -> fano-factor computed directly as variance / mean of spike counts over trials 0-500 ms after the stimulus onset, please note that in the paper we compute fano-factors slightly differently, please refer to the paper
fano_bsl -> baseline fano-factor
boot_fano -> bootstrapped fano-factors 

## Code/Software
The code that is used in our paper lives at 
https://github.com/nurminenlab/variability-sizetuning-analysis

This is an ongoing project so please check in there for the latest versions. At this point the codes are a bit of a mess but I will make them more accessible as we get toward a more final version of the paper.
