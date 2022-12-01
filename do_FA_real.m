% loads 1ms binned spike-counts for neural preselected populations
% and performs gpfa on them and saves results
addpath(genpath('C:\Users\lonurmin\Desktop\code\gpfa_v0203'));
addpath(genpath('C:\Users\lonurmin\Desktop\code\npy-matlab'));

F_dir = 'C:\Users\lonurmin\Desktop\CorrelatedVariability\results\paper_v9\MK-MU\GPFA-long-rasters\';
penetrations = {'MK366P1','MK366P3','MK366P8','MK374P1','MK374P2'};

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
    bls = dir([F_dir,penetrations{i},'*bsl*']);
    C  = [];
    rr = [];
    C_bsl  = [];
    rr_bsl = [];   
    

    stim_ind = 0;
    for stim = 1:length(fls)
        stim_ind = stim_ind + 1;
        rindx = rindx + 1;
        spkR = readNPY([F_dir,fls(stim).name]);
        bslR = readNPY([F_dir,bls(stim).name]);       
        if stim_ind == 1
            netvariance_SG = NaN * ones(sum(layers==1),length(fls));
            netvariance_G = NaN * ones(sum(layers==2),length(fls));
            netvariance_IG = NaN * ones(sum(layers==3),length(fls));
            
            mean_response_SG = NaN * ones(sum(layers==1),length(fls));
            mean_response_G  = NaN * ones(sum(layers==2),length(fls));
            mean_response_IG = NaN * ones(sum(layers==3),length(fls));

            bsl_netvariance_SG = NaN * ones(sum(layers==1),length(bls));
            bsl_netvariance_G = NaN * ones(sum(layers==2),length(bls));
            bsl_netvariance_IG = NaN * ones(sum(layers==3),length(bls));
            
            bsl_mean_response_SG = NaN * ones(sum(layers==1),length(bls));
            bsl_mean_response_G  = NaN * ones(sum(layers==2),length(bls));
            bsl_mean_response_IG = NaN * ones(sum(layers==3),length(bls));
        end
        
        % layers loop
        for l = 1:3
            for tr = 1:size(spkR,3)
                D(tr).spikes = [spkR(layers==l,first_tp:last_tp,tr)]; %#ok<*SAGROW> 
                D(tr).trialId = tr;
                D_bsl(tr).spikes = [bslR(layers==l,:,tr)];
                D_bsl(tr).trialId = tr;
            end          
            % estimate gpfa for evoked responses
            results = neuralTraj(rindx,D);
            CC = diag(results.kern(1).estParams.L*results.kern(1).estParams.L');
            R  = diag(results.kern(1).estParams.Ph);
            spk = results.kern(1).estParams.d;
            pause(0.1);
            rmdir('mat_results','s')
            
            bsl_results = neuralTraj(rindx,D_bsl);
            bsl_CC = diag(bsl_results.kern(1).estParams.L*bsl_results.kern(1).estParams.L');
            bsl_R  = diag(bsl_results.kern(1).estParams.Ph);
            bsl_spk = bsl_results.kern(1).estParams.d;
            pause(0.1);
            rmdir('mat_results','s')

            if l == 1
                netvariance_SG(:,stim_ind) = CC;                
                mean_response_SG(:,stim_ind) = spk;
                bsl_netvariance_SG(:,stim_ind) = bsl_CC;                
                bsl_mean_response_SG(:,stim_ind) = bsl_spk;
            elseif l == 2
                netvariance_G(:,stim_ind)  = CC;
                mean_response_G(:,stim_ind)  = spk;
                bsl_netvariance_G(:,stim_ind)  = bsl_CC;
                bsl_mean_response_G(:,stim_ind)  = bsl_spk;
            else
                netvariance_IG(:,stim_ind) = CC;
                mean_response_IG(:,stim_ind)  = spk;
                bsl_netvariance_IG(:,stim_ind) = bsl_CC;
                bsl_mean_response_IG(:,stim_ind)  = bsl_spk;

            end            

        end

        clear('D','D_bsl','spkR','bslR');

    end


    writeNPY(netvariance_SG,['netvariance_SG',penetrations{i},'.npy']);
    writeNPY(netvariance_G,['netvariance_G',penetrations{i},'.npy']);
    writeNPY(netvariance_IG,['netvariance_IG',penetrations{i},'.npy']);

    writeNPY(bsl_netvariance_SG,['bsl_netvariance_SG',penetrations{i},'.npy']);
    writeNPY(bsl_netvariance_G,['bsl_netvariance_G',penetrations{i},'.npy']);
    writeNPY(bsl_netvariance_IG,['bsl_netvariance_IG',penetrations{i},'.npy']);

    if size(netvariance_SG,2 ) == 18
        netvariance_SG = [nan*ones(size(netvariance_SG,1),1), netvariance_SG]; %#ok<*AGROW> 
        netvariance_G  = [nan*ones(size(netvariance_G,1),1), netvariance_G];
        netvariance_IG = [nan*ones(size(netvariance_IG,1),1), netvariance_IG];

        mean_response_SG = [nan*ones(size(mean_response_SG,1),1), mean_response_SG];
        mean_response_G  = [nan*ones(size(mean_response_G,1),1), mean_response_G];
        mean_response_IG = [nan*ones(size(mean_response_IG,1),1), mean_response_IG];

        bsl_netvariance_SG = [nan*ones(size(bsl_netvariance_SG,1),1), bsl_netvariance_SG]; %#ok<*AGROW> 
        bsl_netvariance_G  = [nan*ones(size(bsl_netvariance_G,1),1), bsl_netvariance_G];
        bsl_netvariance_IG = [nan*ones(size(bsl_netvariance_IG,1),1), bsl_netvariance_IG];

        bsl_mean_response_SG = [nan*ones(size(bsl_mean_response_SG,1),1), bsl_mean_response_SG];
        bsl_mean_response_G  = [nan*ones(size(bsl_mean_response_G,1),1), bsl_mean_response_G];
        bsl_mean_response_IG = [nan*ones(size(bsl_mean_response_IG,1),1), bsl_mean_response_IG];
    end

    netvariance_all_SG = [netvariance_all_SG;netvariance_SG];
    netvariance_all_G = [netvariance_all_G;netvariance_G];
    netvariance_all_IG = [netvariance_all_IG;netvariance_IG];
    
    mean_response_all_SG = [mean_response_all_SG;mean_response_SG];
    mean_response_all_G = [mean_response_all_G;mean_response_G];
    mean_response_all_IG = [mean_response_all_IG;mean_response_IG];

    bsl_netvariance_all_SG = [bsl_netvariance_all_SG;bsl_netvariance_SG];
    bsl_netvariance_all_G = [bsl_netvariance_all_G;bsl_netvariance_G];
    bsl_netvariance_all_IG = [bsl_netvariance_all_IG;bsl_netvariance_IG];
    
    bsl_mean_response_all_SG = [bsl_mean_response_all_SG;bsl_mean_response_SG];
    bsl_mean_response_all_G = [bsl_mean_response_all_G;bsl_mean_response_G];
    bsl_mean_response_all_IG = [bsl_mean_response_all_IG;bsl_mean_response_IG];

end

diams = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26];

mean_response_all_SG = mean_response_all_SG./0.08;
mean_response_all_IG = mean_response_all_IG./0.08;
mean_response_all_G  = mean_response_all_G./0.08;

writeNPY(netvariance_all_SG,'netvariance_all_SG.npy');
writeNPY(mean_response_all_SG,'mean_response_all_SG.npy');
writeNPY(bsl_netvariance_all_SG,'bsl_netvariance_all_SG.npy');
writeNPY(bsl_mean_response_all_SG,'bsl_mean_response_all_SG.npy');

writeNPY(netvariance_all_G,'netvariance_all_G.npy');
writeNPY(mean_response_all_G,'mean_response_all_G.npy');
writeNPY(bsl_netvariance_all_G,'bsl_netvariance_all_G.npy');
writeNPY(bsl_mean_response_all_G,'bsl_mean_response_all_G.npy');

writeNPY(netvariance_all_IG,'netvariance_all_IG.npy');
writeNPY(mean_response_all_IG,'mean_response_all_IG.npy');
writeNPY(bsl_netvariance_all_IG,'bsl_netvariance_all_IG.npy');
writeNPY(bsl_mean_response_all_IG,'bsl_mean_response_all_IG.npy');

