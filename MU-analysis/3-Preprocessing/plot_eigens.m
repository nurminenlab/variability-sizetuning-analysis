function plot_eigens(eigens)

for i = 1:size(eigens,3)
    EG = nanmean(eigens(:,:,i),1); %#ok<NANMEAN> 
    SE = nanstd(eigens(:,:,i),1)./sqrt(sum(~isnan(eigens(:,:,i)),1)); %#ok<NANSTD> 
    figure    
    errorbar(1:length(EG), EG,SE);
    set(gca,'XLim',[0, 10]);
end

