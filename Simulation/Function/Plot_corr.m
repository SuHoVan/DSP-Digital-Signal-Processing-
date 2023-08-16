clear all; close all; clc;
%== Plot SNR==
data_corr = load('D:\EEG\Correlation_final.csv');
t = 10:1:30;
data_corr = data_corr';

corr = figure;
hold on; grid on;
%title('Time of Algorithms');

semilogy(t,data_corr(1,:),'k-o','linewidth',1.5,'markersize',4);
semilogy(t,data_corr(2,:),'r-s','linewidth',1.5,'markersize',4); 
semilogy(t,data_corr(3,:),'k-.d','linewidth',1.5,'markersize',4);
semilogy(t,data_corr(4,:),'r-.h','linewidth',1.5,'markersize',4);
semilogy(t,data_corr(5,:),'b-.*','linewidth',1.5,'markersize',4);

xlabel('SNR Noises [dB]','Interpreter','latex',"FontSize",11);ylabel('Correlation coefficient','Interpreter','latex',"FontSize",11);

legend('$\mathrm{\widehat{EEG}}$','$\mathrm{\widehat{EEG}_{denoise}}$','$\mathrm{\widehat{ECG}}$', ...
    '$\mathrm{\widetilde{ECG}}$','$\mathrm{\widetilde{ECG}_{denoise}}$', 'Interpreter','latex',"FontSize",11);
ylim([0,1])
