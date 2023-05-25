## Analysis scripts for Nurminen, Bijanzadeh and Angelucci 2023
https://doi.org/10.1101/2023.01.17.524397

To replicate the results in the paper, please follow the workflow below. 

If you would like to use our spike-sorting or multi-unit thresholding, please follow the instructions below from point 4. If you would like to perform the analysis from the scratch, please start from point 1.

1. Download raw data files from ...
2. For single-unit analysis, run spike-sorting with Kilosort. For multi-units, first re-threshold the data with ln_rethreshold.m from https://github.com/nurminenlab/Preprocessing. There are dependencies on an old Blackrock Microsystem's NPMK library and modified versions of their functions. We provide access to the legacy NPMK version at https://github.com/nurminenlab/NPMK
3. For spike-sorted data, please run KilosortExtractSpikeCountsRasters_nolasers.m from https://github.com/nurminenlab/Preprocessing. This creates a trial based representation of spike counts and rasters for all stimulus conditions. For multi-unit analysis, run ExtractSpikeCountsRasters_nolasers.m from https://github.com/nurminenlab/Preprocessing. These scripts need to be run for all datafiles separately.
4. 
