% loads 1ms binned spike-counts for neural preselected populations
% and performs gpfa on them and saves results
addpath(genpath('C:\Users\lonurmin\Desktop\code\gpfa_v0203'));
addpath(genpath('C:\Users\lonurmin\Desktop\code\npy-matlab'));

F_dir = 'C:\Users\lonurmin\Desktop\CorrelatedVariability\results\paper_v9\MK-MU\GPFA-long-rasters\';
penetrations = {'MK366P1','MK366P3','MK366P8','MK374P1','MK374P2'};

SG_units = NaN * ones(1,length(penetrations));
G_units = NaN * ones(1,length(penetrations));
IG_units = NaN * ones(1,length(penetrations));

for i = 1:length(penetrations)        
    layer_fl = dir([F_dir,'layers_',penetrations{i},'.npy']);
    layers   = readNPY([F_dir,layer_fl.name]);
    SG_units(i) = sum(layers==1);
    G_units(i)  = sum(layers==2);
    IG_units(i) = sum(layers==3);
end

%diams = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.4, 3, 3.5, 5, 10, 15, 20, 26];
