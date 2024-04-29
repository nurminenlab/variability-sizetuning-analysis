% loads 1ms binned spike-counts for neural preselected populations
% and performs gpfa on them and saves results
addpath(genpath('C:\Users\lonurmin\Desktop\code\gpfa_v0203'));
addpath(genpath('C:\Users\lonurmin\Desktop\code\npy-matlab'));
base_dir = 'C:\Users\lonurmin\Desktop\AnalysisScripts\VariabilitySizeTuning\variability-sizetuning-analysis\MU-analysis\2-PrecomputedAnalysis\';
F_dir = 'C:\Users\lonurmin\Desktop\CorrelatedVariability\results\paper_v9\MK-MU\GPFA-long-rasters\';
penetrations = {'MK366P1','MK366P3','MK366P8','MK374P1','MK374P2'};

SG_animal = [];
G_animal  = [];
IG_animal = [];

for i = 1:length(penetrations)
    layer_fl = dir([F_dir,'layers_',penetrations{i},'.npy']);
    layers   = readNPY([F_dir,layer_fl.name]);
    
    SG_animal = [SG_animal;repmat(penetrations{i}(3:5),sum(layers == 1),1)]; %#ok<AGROW> 
    G_animal  = [G_animal;repmat(penetrations{i}(3:5),sum(layers == 2),1)]; %#ok<AGROW> 
    IG_animal = [IG_animal;repmat(penetrations{i}(3:5),sum(layers == 3),1)]; %#ok<AGROW> 
end

% animal IDs for each unit
writeNPY(str2num(SG_animal),[base_dir,'SG_unit_animals.npy']); %#ok<*ST2NM> 
writeNPY(str2num(G_animal),[base_dir,'G_unit_animals.npy']);
writeNPY(str2num(IG_animal),[base_dir,'IG_unit_animals.npy']);

