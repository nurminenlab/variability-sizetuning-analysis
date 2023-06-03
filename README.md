## Analysis scripts for Nurminen, Bijanzadeh and Angelucci 2023
https://doi.org/10.1101/2023.01.17.524397

To replicate the results in the paper, please follow the workflow below. 

Many of the scripts depend on several standard python packages. Please refer to the individual files for details, but installing numpy, matplotlib, pandas, scipy, statsmodels, pickle, and pytables should be pretty much everything you need.

If you would like to use our spike-sorting or multi-unit thresholding, please follow the instructions below from point 4. If you would like to perform the analysis from the scratch, please start from point 1.

1. Download raw data files from ...
2. **Spike sorting and multi-unit thresholding** <br /> For single-unit analysis, run spike-sorting with Kilosort. For multi-units, first re-threshold the data with ln_rethreshold.m from https://github.com/nurminenlab/Preprocessing. There are dependencies on an old Blackrock Microsystem's NPMK library and modified versions of their functions. We provide access to the legacy NPMK version at https://github.com/nurminenlab/NPMK
3. **Trial based representation of stimulus conditions** <br /> For spike-sorted data, please run KilosortExtractSpikeCountsRasters_nolasers.m from https://github.com/nurminenlab/Preprocessing. This creates a trial based representation of spike counts and rasters for all stimulus conditions. For multi-unit analysis, run ExtractSpikeCountsRasters_nolasers.m from https://github.com/nurminenlab/Preprocessing. These scripts need to be run for all datafiles separately.
4. **Reformat to datatable** <br /> From https://github.com/nurminenlab/Preprocessing, run Kilosorted2Pytable_macaque.py for single-unit analysis or MUA2Pytable_macaque.py for multi-unit analysis. These scripts require information about layers. The data folders include our estimates of the borders of cortical layers. You can also estimate these from the data using CSD. Just average the evoked LFPs across all the grating diameters and run CSD. You will also need the file penetrationinfo_macaque.csv at the root of this repo.
5. **Data selection** <br /> For multi-unit analysis, run select_fano_data_MUA.py. For single-unit analysis, run select_fano_data.py. These scripts produce as outputs, selectedData_MUA_lenient_400ms_macaque_July-2020.pkl, and selectedData_lenient_400ms_macaque_July-2020.pkl, respectively. 
6. **Parameter extraction** <br /> Run extract-paramsTable-fano-nolaser-MUonly.py to fit firing-rate and fano-factor data with DOiG functions and extract the parameters we report in the paper. This step produces as output a CSV files extracted_params-Oct-2022.csv which contains extracted parameters for all recorded multi-units. The script also outputs PSTHs for variance and firing rate. These are stored in files called 'mean_PSTHs_SG-MK-MU-'+month+year+'.pkl' and 'vari_PSTHs_SG-MK-MU-'+month+year+'.pkl'. These files are produced for all layers. *Note to self. Modify this file for single-units analysis*.
8. **Analyses for variability size tuning, Figures 1-2.** <br /> 
    - Generate example plots by running Figure1_examples_SG.py, Figure1_examples_G.py, Figure1_examples_IG.py
    - Run generate_quencher_DF.py for statistical testing of surround effects on fano-factor
    - Do population analyses by running Figure2A.py, Figure2B.py, Figure2CDG-top.py, Figure2CD-bottom.py, Figure2E.py, Figure2F.py
9. **Variability amplification, Figure 3.** <br /> 
    - Run Figure3A.py to plot the example amplifiers.
    - Run Figure3B.py to plot mean PSTHs across the layers
    - To prepare intermediate files for population analysis of variability amplification, run prepare_amplification_analysis_division.py, you can skip this step if you wish to use our precomputed files
    - Run Figure3C.py, Figure3D.py, Figure3E.py
10. **Shared variance examples. Figure 4.** <br /> 
    - To plot the rasters, run Figure4ABC-rasters.py
    - To plot the example network covariance matrices, run Figure4ABC_covariances.m
    - Run Figure4ABC-size-tuning.py
11. **Shared variance population data. Figure 5.** <br /> 









