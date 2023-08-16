clear all; close all; clc;
%== Plot SNR==
data_snr = load('D:\EEG\SNR.csv');
t = 10:1:20;
data_snr = data_snr';

snr = figure();
hold on; grid on;
%title('Time of Algorithms');

semilogy(t,data_snr(1,:),'k--d','linewidth',1.5,'markersize',4);
semilogy(t,data_snr(2,:),'k-o','linewidth',1.5,'markersize',4); 

xlabel('SNR Noises [dB]','Interpreter','latex',"FontSize",11);ylabel('SNR ECG [dB]','Interpreter','latex',"FontSize",11);
legend('$\mathrm{\widehat{ECG}}$','$\mathrm{\widetilde{ECG}}$','Interpreter','latex',"FontSize",11);
exportgraphics(snr, 'SNR.pdf', 'ContentType','vector');