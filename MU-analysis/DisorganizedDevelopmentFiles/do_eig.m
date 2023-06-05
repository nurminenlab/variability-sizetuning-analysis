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
    
    if length(fls) == 19
        fls = fls([2,4,19]);    
    else
        fls = fls([1,3,18]);    
    end

    stim_ind = 0;
    for stim = 1:length(fls) 
        stim_ind = stim_ind + 1;
        rindx = rindx + 1;
        spkR = readNPY([F_dir,fls(stim).name]);
        bslR = readNPY([F_dir,bls(stim).name]);       
        if stim_ind == 1
            
            SG_eigs = ones(sum(layers==1),3);
            G_eigs  = ones(sum(layers==2),3);
            IG_eigs = ones(sum(layers==3),3);

        end

        SG_spkR = squeeze(sum(spkR(layers==1,first_tp:last_tp,:),2))';
        G_spkR  = squeeze(sum(spkR(layers==2,first_tp:last_tp,:),2))';
        IG_spkR = squeeze(sum(spkR(layers==3,first_tp:last_tp,:),2))';        

        C = cov(SG_spkR);
        SG_eigs(:,stim) = eig(C);
        C = cov(G_spkR);
        G_eigs(:,stim) = eig(C);
        C = cov(IG_spkR);
        IG_eigs(:,stim) = eig(C);

    end
    figure('Name',penetrations{i})
    subplot(3,2,1)
    hold on    
    plot(flipud(SG_eigs(:,1)),'ro-');
    plot(flipud(SG_eigs(:,2)),'ko-');    

    subplot(3,2,2)
    hold on    
    plot(flipud(SG_eigs(:,2)),'ko-');
    plot(flipud(SG_eigs(:,3)),'bo-');   
    
    subplot(3,2,3)
    hold on    
    plot(flipud(G_eigs(:,1)),'ro-');
    plot(flipud(G_eigs(:,2)),'ko-');    

    subplot(3,2,4)
    hold on    
    plot(flipud(G_eigs(:,2)),'ko-');
    plot(flipud(G_eigs(:,3)),'bo-');   

    subplot(3,2,5)
    hold on    
    plot(flipud(IG_eigs(:,1)),'ro-');
    plot(flipud(IG_eigs(:,2)),'ko-');    

    subplot(3,2,6)
    hold on    
    plot(flipud(IG_eigs(:,2)),'ko-');
    plot(flipud(IG_eigs(:,3)),'bo-');   

end

        

        

