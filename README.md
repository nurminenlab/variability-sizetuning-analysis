## Analysis scripts for Nurminen, Bijanzadeh and Angelucci 2023
https://doi.org/10.1101/2023.01.17.524397

To replicate the results in the paper, please follow the workflow below. 

If you would like to use our spike-sorting or multi-unit thresholding, please follow the instructions below from point 4. If you would like to perform the analysis from the scratch, please start from point 1.

1. Download raw data files from ...
2. Run spike-sorting for single-unit analysis or re-thresholding for multi-units. For re-thresholding, use ln_rethreshold.m from our Preprocessing repo https://github.com/nurminenlab/Preprocessing ln_rethreshold.m depends on Blackrock Microsystem's NPMK library and modified versions of their functions. We provide access to the legacy NPMK version at https://github.com/nurminenlab/NPMK