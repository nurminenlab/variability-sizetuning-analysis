% create "editable" eps PSTHs

mat_dir = 'C:\Users\lonurmin\Desktop\CorrelatedVariability\results\paper_v9\MK-MU\PSTHmats\';


%%
% SG 
load([mat_dir, 'PSTHsSG_stim_2.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;

t2 = [t, fliplr(t)];

subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceColor',[1,0,0]);
hold on 
plot(t,fano_PSTH_RF,'r-','LineWidth',2,'Color',[0.5,0,0])
plot([-100, 600], [mean(fano_PSTH_RF(1:100)), mean(fano_PSTH_RF(1:100))],'r--','LineWidth',2)
axis([-100, 600, 0, 12.5])
% FR

PSTH_RF = mean(PSTH_RF,1);
PSTH_RF_ub = PSTH_RF + PSTH_RF_SD;
PSTH_RF_lb = PSTH_RF - PSTH_RF_SD;

yyaxis right
fill(t2,[PSTH_RF_lb, fliplr(PSTH_RF_ub)],'k');
hold on 
plot(t,PSTH_RF,'k--','LineWidth',2,'Color',[0.5,0.5,0.5])
plot([-100, 600], [mean(PSTH_RF(1:100)), mean(PSTH_RF(1:100))],'k--','LineWidth',2)
axis([-100, 600, 0, 250])

box off
set(gca,'TickDir','out')

% large
load([mat_dir, 'PSTHsSG_stim_18.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;
t2 = [t, fliplr(t)];

subplot(2,1,2)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r');
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2,'Color',[0.5,0,0])
plot([-100, 600], [mean(fano_PSTH_RF(1:100)), mean(fano_PSTH_RF(1:100))],'r--','LineWidth',2)
axis([-100, 600, 0, 12.5])

PSTH_RF = mean(PSTH_RF,1);
PSTH_RF_ub = PSTH_RF + PSTH_RF_SD;
PSTH_RF_lb = PSTH_RF - PSTH_RF_SD;

yyaxis right
fill(t2,[PSTH_RF_lb, fliplr(PSTH_RF_ub)],'k');
hold on
plot(t,PSTH_RF,'k--','LineWidth',2,'Color',[0.5,0.5,0.5])
plot([-100, 600], [mean(PSTH_RF(1:100)), mean(PSTH_RF(1:100))],'k--','LineWidth',2)
axis([-100, 600, 0, 250])

box off
set(gca,'TickDir','out')

% truncated FF
load([mat_dir, 'truncated_PSTHsSG_stim_2.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;
t2 = [t, fliplr(t)];

figure(2)
subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
axis([50, 350, 0, 4])
% FR

% large
load([mat_dir, 'truncated_PSTHsSG_stim_18.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;

subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
axis([50, 350, 0, 4])


%%
figure(3)
% G 
load([mat_dir, 'PSTHsG_stim_4.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;

t2 = [t, fliplr(t)];

subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
plot([-100, 600], [mean(fano_PSTH_RF(1:100)), mean(fano_PSTH_RF(1:100))],'r--','LineWidth',2)
axis([-100, 600, 0, 4])
% FR

PSTH_RF = mean(PSTH_RF,1);
PSTH_RF_ub = PSTH_RF + PSTH_RF_SD;
PSTH_RF_lb = PSTH_RF - PSTH_RF_SD;

yyaxis right
fill(t2,[PSTH_RF_lb, fliplr(PSTH_RF_ub)],'k','FaceAlpha',0.5);
hold on 
plot(t,PSTH_RF,'k--','LineWidth',2)
plot([-100, 600], [mean(PSTH_RF(1:100)), mean(PSTH_RF(1:100))],'k--','LineWidth',2)
axis([-100, 600, 0, 155])

box off
set(gca,'TickDir','out')

% large
load([mat_dir, 'PSTHsG_stim_18.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;
t2 = [t, fliplr(t)];

subplot(2,1,2)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
plot([-100, 600], [mean(fano_PSTH_RF(1:100)), mean(fano_PSTH_RF(1:100))],'r--','LineWidth',2)
axis([-100, 600, 0, 4])

PSTH_RF = mean(PSTH_RF,1);
PSTH_RF_ub = PSTH_RF + PSTH_RF_SD;
PSTH_RF_lb = PSTH_RF - PSTH_RF_SD;

yyaxis right
fill(t2,[PSTH_RF_lb, fliplr(PSTH_RF_ub)],'k','FaceAlpha',0.5);
hold on 
plot(t,PSTH_RF,'k--','LineWidth',2)
plot([-100, 600], [mean(PSTH_RF(1:100)), mean(PSTH_RF(1:100))],'k--','LineWidth',2)
axis([-100, 600, 0, 155])
box off
set(gca,'TickDir','out')

% truncated FF
load([mat_dir, 'truncated_PSTHsG_stim_4.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;
t2 = [t, fliplr(t)];

figure(4)
subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
axis([50, 350, 0, 3])
% FR

% large
load([mat_dir, 'truncated_PSTHsG_stim_18.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;

subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
axis([50, 350, 0, 3])
box off
set(gca,'TickDir','out')

%%
figure(5)
% IG 
load([mat_dir, 'PSTHsIG_stim_4.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;

t2 = [t, fliplr(t)];

subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
plot([-100, 600], [mean(fano_PSTH_RF(1:100)), mean(fano_PSTH_RF(1:100))],'r--','LineWidth',2)
axis([-100, 600, 0, 4])
% FR

PSTH_RF = mean(PSTH_RF,1);
PSTH_RF_ub = PSTH_RF + PSTH_RF_SD;
PSTH_RF_lb = PSTH_RF - PSTH_RF_SD;

yyaxis right
fill(t2,[PSTH_RF_lb, fliplr(PSTH_RF_ub)],'k','FaceAlpha',0.5);
hold on 
plot(t,PSTH_RF,'k--','LineWidth',2)
plot([-100, 600], [mean(PSTH_RF(1:100)), mean(PSTH_RF(1:100))],'k--','LineWidth',2)
axis([-100, 600, 0, 155])

box off
set(gca,'TickDir','out')

% large
load([mat_dir, 'PSTHsIG_stim_18.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;
t2 = [t, fliplr(t)];

subplot(2,1,2)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
plot([-100, 600], [mean(fano_PSTH_RF(1:100)), mean(fano_PSTH_RF(1:100))],'r--','LineWidth',2)
axis([-100, 600, 0, 4])

PSTH_RF = mean(PSTH_RF,1);
PSTH_RF_ub = PSTH_RF + PSTH_RF_SD;
PSTH_RF_lb = PSTH_RF - PSTH_RF_SD;

yyaxis right
fill(t2,[PSTH_RF_lb, fliplr(PSTH_RF_ub)],'k','FaceAlpha',0.5);
hold on 
plot(t,PSTH_RF,'k--','LineWidth',2)
plot([-100, 600], [mean(PSTH_RF(1:100)), mean(PSTH_RF(1:100))],'k--','LineWidth',2)
axis([-100, 600, 0, 155])
box off
set(gca,'TickDir','out')

% truncated FF
load([mat_dir, 'truncated_PSTHsIG_stim_4.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;
t2 = [t, fliplr(t)];

figure(6)
subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
axis([50, 350, 0, 3])
% FR

% large
load([mat_dir, 'truncated_PSTHsIG_stim_18.mat']);
fano_PSTH_RF = mean(fano_PSTH_RF,1);
fano_PSTH_RF_ub = fano_PSTH_RF + fano_PSTH_RF_SD;
fano_PSTH_RF_lb = fano_PSTH_RF - fano_PSTH_RF_SD;

subplot(2,1,1)
fill(t2,[fano_PSTH_RF_lb, fliplr(fano_PSTH_RF_ub)],'r','FaceAlpha',0.5);
hold on 
plot(t,fano_PSTH_RF,'r--','LineWidth',2)
axis([50, 350, 0, 3])
box off
set(gca,'TickDir','out')