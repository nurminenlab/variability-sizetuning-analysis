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

% SG = 1, G = 2, IG = 3
layer_ind = 1;

anal_duration = 400;
first_tp = 150;
last_tp  = first_tp + anal_duration;
for i = 1:length(penetrations)
    
    layer_fl = dir([F_dir,'layers_',penetrations{i},'.npy']);
    layers   = readNPY([F_dir,layer_fl.name]);
    
    fls = dir([F_dir,penetrations{i},'*stim*']);
    fls = fls([1,4,19]); % stimulus diameters

    stim_ind = 0;
    CC_all = ones(sum(layers==layer_ind),sum(layers==layer_ind),length(fls));
    for stim = 1:length(fls)
        stim_ind = stim_ind + 1;
        rindx = rindx + 1;
        spkR = readNPY([F_dir,fls(stim).name]);        

        % layers loop
        for l = 1            
            for tr = 1:size(spkR,3)
                D(tr).spikes = [spkR(layers==layer_ind,first_tp:last_tp,tr)]; %#ok<*SAGROW> 
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

