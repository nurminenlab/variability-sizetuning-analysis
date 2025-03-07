% loads 1ms binned spike-counts for neural preselected populations
% and performs gpfa on them and saves results
addpath(genpath('C:\Users\lonurmin\Desktop\code\gpfa_v0203'));
addpath(genpath('C:\Users\lonurmin\Desktop\code\npy-matlab'));

F_dir = 'C:\Users\lonurmin\Desktop\CorrelatedVariability\results\paper_v9\MK-MU\GPFA-long-rasters\';
% penetrations = {'MK366P1','MK366P3','MK366P8','MK374P1','MK374P2'};

penetrations = {'MK374P2'};

stimulus_used = 2;

n_factors = 3;
n_diameters = 19;

SG_RMSEs = NaN * ones(length(penetrations),n_diameters,4);
G_RMSEs  = NaN * ones(length(penetrations),n_diameters,3);
IG_RMSEs = NaN * ones(length(penetrations),n_diameters, 7);

rindx = 0;
C_final  = []
rr_final = []

n_penetrations = 5;
n_diameters    = 19;
n_eigen        = 20;

n_SG = 0;
n_G = 0;
n_IG = 0;

anal_duration = 400;
first_tp = 150;
last_tp  = first_tp + anal_duration;
for i = 1:length(penetrations)    
    
    layer_fl = dir([F_dir,'layers_',penetrations{i},'.npy']);
    layers   = readNPY([F_dir,layer_fl.name]);
    
    fls = dir([F_dir,penetrations{i},'*stim*']);
    bls = dir([F_dir,penetrations{i},'*bsl*']);
    
    fls = fls(stimulus_used);
    bls = bls(stimulus_used);

    C  = [];
    rr = [];
    C_bsl  = [];
    rr_bsl = [];       

    stim_ind = 0;
    for stim = 1:length(fls)
        clear('D','D_bsl','spkR','bslR');

        stim_ind = stim_ind + 1;
        rindx = rindx + 1;
        spkR = readNPY([F_dir,fls(stim).name]);
        bslR = readNPY([F_dir,bls(stim).name]);                  
        
        keyboard

        % layers loop, for cross-validating FA dimensionality, we do the
        % analysis layer-by-layer
        for l = 1:3
            if l == 1
                n_factors = 4;
            elseif l == 2
                n_factors = 3;
            else
                n_factors = 7;
            end           


            for f = 1:n_factors            

                for tr = 1:size(spkR,3)
                    D(tr).spikes = [spkR(layers==l,first_tp:last_tp,tr)]; %#ok<*SAGROW> 
                    D(tr).trialId = tr;
                    D_bsl(tr).spikes = [bslR(layers==l,:,tr)];
                    D_bsl(tr).trialId = tr;
                end          
                % estimate gpfa for evoked responses
                [results,RMSE] = neuralTraj_CV(rindx,D,f,5);
                pause(0.1);
                
                if length(fls) == 18
                    si = stim +1;
                else
                    si = stim;
                end
                if l == 1
                    SG_RMSEs(i,si,f) = mean(RMSE);
                elseif l == 2
                    G_RMSEs(i,si,f) = mean(RMSE);
                else
                    IG_RMSEs(i,si,f) = mean(RMSE);
                end              
                
                rmdir('mat_results','s');
                
    %             bsl_results = neuralTraj(rindx,D_bsl);
    %             bsl_CC = diag(bsl_results.kern(1).estParams.L*bsl_results.kern(1).estParams.L');
    %             bsl_R  = diag(bsl_results.kern(1).estParams.Ph);
    %             bsl_spk = bsl_results.kern(1).estParams.d;
    %             pause(0.1);
    %             rmdir('mat_results','s')                            
    
            end                  
        end
    end
end

save_root = 'C:\Users\lonurmin\Desktop\AnalysisScripts\VariabilitySizeTuning\variability-sizetuning-analysis\MU-analysis\2-PrecomputedAnalysis\';
save([save_root,'FA_RMSEs.mat'],'SG_RMSEs','G_RMSEs','IG_RMSEs');

%diams = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26];
