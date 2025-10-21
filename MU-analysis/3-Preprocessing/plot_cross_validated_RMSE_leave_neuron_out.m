save_root = 'C:\Users\lonurmin\Desktop\AnalysisScripts\VariabilitySizeTuning\variability-sizetuning-analysis\MU-analysis\2-PrecomputedAnalysis\';
load([save_root,'FA_RMSEs.mat'])


SG_SE = squeeze(nanstd(SG_RMSEs,1))./sqrt(5); %#ok<NANSTD> 
G_SE  = squeeze(nanstd(G_RMSEs,1))./sqrt(5); %#ok<NANSTD> 
IG_SE = squeeze(nanstd(IG_RMSEs,1))./sqrt(5); %#ok<NANSTD>

SG_all = SG_RMSEs;
G_all  = G_RMSEs;
IG_all = IG_RMSEs;

SG_RMSEs = squeeze(nanmean(SG_RMSEs,1)); %#ok<NANMEAN> 
G_RMSEs  = squeeze(nanmean(G_RMSEs,1)); %#ok<NANMEAN> 
IG_RMSEs = squeeze(nanmean(IG_RMSEs,1)); %#ok<NANMEAN> 

figure('Name','SG')
SG_stats = ones(1,19);
SG_p = ones(1,19);
for i = 1:19
    subplot(4,5,i)
    errorbar(SG_RMSEs(i,:),SG_SE(i,:),'ko-');
    [SG_stats(i),SG_p(i)] = ttest2(SG_all(:,i,1),SG_all(:,i,2));
end

figure('Name','G')
G_stats = ones(1,19);
for i = 1:19
    subplot(4,5,i)
    errorbar(G_RMSEs(i,:),G_SE(i,:),'ko-');
    G_stats(i) = ttest2(G_all(:,i,1),G_all(:,i,2));
end

figure('Name','IG')
IG_stats = ones(1,19);
for i = 1:19
    subplot(4,5,i)
    errorbar(IG_RMSEs(i,:),IG_SE(i,:),'ko-');
    IG_stats(i) = ttest2(IG_all(:,i,1),IG_all(:,i,2));
end