% loads 1ms binned spike-counts for neural preselected populations
% and performs gpfa on them and saves results
addpath(genpath('C:\Users\lonurmin\Desktop\code\gpfa_v0203'));
addpath(genpath('C:\Users\lonurmin\Desktop\code\npy-matlab'));

F_dir = 'C:\Users\lonurmin\Desktop\CorrelatedVariability\results\paper_v9\MK-MU\GPFA-long-rasters\';
%penetrations = {'MK366P1','MK366P3','MK366P8','MK374P1','MK374P2'};
penetrations = {'MK374P1'};
rindx = 0;
C_final  = []
rr_final = []

netvariance_all_SG = [];
netvariance_all_G  = [];
netvariance_all_IG = [];

mean_response_all_SG = [];
mean_response_all_G  = [];
mean_response_all_IG = [];

privatevariance_all_SG = [];
privatevariance_all_G  = [];
privatevariance_all_IG = [];

% bsl
bsl_netvariance_all_SG = [];
bsl_netvariance_all_G  = [];
bsl_netvariance_all_IG = [];

bsl_mean_response_all_SG = [];
bsl_mean_response_all_G  = [];
bsl_mean_response_all_IG = [];

bsl_privatevariance_all_SG = [];
bsl_privatevariance_all_G  = [];
bsl_privatevariance_all_IG = [];

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
    fls = fls([1,4,19]);

    stim_ind = 0;
    CC_all = ones(sum(layers==3),sum(layers==3),length(fls));
    for stim = 1:length(fls)
        stim_ind = stim_ind + 1;
        rindx = rindx + 1;
        spkR = readNPY([F_dir,fls(stim).name]);        

        % layers loop
        for l = 1            
            for tr = 1:size(spkR,3)
                D(tr).spikes = [spkR(layers==3,first_tp:last_tp,tr)]; %#ok<*SAGROW> 
                D(tr).trialId = tr;                
            end          
            % estimate gpfa for evoked responses
            results = neuralTraj(rindx,D);
            CC = results.kern(1).estParams.L*results.kern(1).estParams.L';            
            CC_all(:,:,stim) = CC;
            rmdir('mat_results','s')
        end
        clear('D','D_bsl','spkR','bslR');
    end
end

%writeNPY(CC_all, 'FA_covariances_IG.npy');

subplot(2,2,1)
image(101*CC_all(:,:,1)); colormap("jet"); colorbar;
axis square off

subplot(2,2,2)
image(101*CC_all(:,:,2)); colormap("jet"); colorbar;
axis square off

subplot(2,2,3)
image(101*CC_all(:,:,3)); colormap("jet"); colorbar;
axis square off

