clear all; close all; clc;

load_data = load("D:\EEG\data_filtered.csv", '-ascii');
Fs = 250;                                   % Sample Freq [Hz]
Ts = 1/Fs;                                  % Sample Rate [second]
range=60;% start [second]
t = 1:length(load_data);
N = length(t);
f_axis = (0:N/2-1)*Fs/N;

%% Norm and mix
eeg1 = load_data(1,:);
eeg2 = load_data(2,:);
chieu2= load_data(4,:);

%---------------------
figure
x1 = subplot(3,1,1);
plot(t,eeg1);xlabel('EEG 1');title("Source data"); grid on;
% ylim([-0.25 1.5]);
x2 = subplot(3,1,2);
plot(t,eeg2);xlabel('EEG 2');grid on;
% ylim([-0.25 1.5]);
x3 = subplot(3,1,3);
plot(t,chieu1);xlabel('ECG');grid on;
% ylim([-0.25 1.5]);
linkaxes([x1,x2,x3],'x');

data = [eeg1;eeg2];

%% ======= ICA 1st ==========
data_ICA = ICA(data);

eeg_ICA = -data_ICA(1,:);
ecg_ICA = data_ICA(2,:);

ecg_ICA = remove_dt(t,ecg_ICA);   %remove detrend of ECG signal


figure
x1 = subplot(2,1,1);
plot(t,eeg_ICA);xlabel('EEG');title("After ICA"); grid on;
x2 = subplot(2,1,2);
plot(t,ecg_ICA);xlabel('ECG');grid on;
linkaxes([x1,x2],'x');


%% ======= Wavelet Denoising 1st ==========
% Remove noise on the estimated EEG signal
eeg_wavelet = preprocess(eeg_ICA);


figure
x1 = subplot(2,1,1);
plot(t,eeg_ICA);title("EEG after ICA"); grid on; 
% plot(t,eeg1,'r');
% ylim([-0.25 1.5]);

x2 = subplot(2,1,2);
plot(t,eeg_wavelet);grid on;title("EEG wavelet denoise")
% plot(t,eeg2,'r');
% ylim([-0.25 1.5]);
linkaxes([x1,x2],'x');

%% ======== SIC process ============
diff1 = eeg1 - eeg_wavelet;
diff2 = eeg2 - eeg_wavelet;


figure
x1 = subplot(2,1,1);
plot(t,diff1);xlabel('Diff1');title("Diff"); grid on;
% ylim([-0.25 1.5]);
x2 = subplot(2,1,2);
plot(t,diff2);xlabel('Diff2');grid on;
% ylim([-0.25 1.5]);
linkaxes([x1,x2],'x');
% 

data2 = [diff1;diff2;ecg_ICA];

%% ======= ICA 2nd ==========
data_ICA2 = ICA(data2);

ecg_ICA2 = remove_dt(t,data_ICA2(1,:)); 
 
% plot
figure
x1 = subplot(3,1,1);
plot(t,data_ICA2(1,:));xlabel('ECG after ICA2');title("ICA2"); grid on;

x2 = subplot(3,1,2);
plot(t,data_ICA2(2,:));xlabel('output 2');grid on;

x3 = subplot(3,1,3);
plot(t,data_ICA2(3,:));xlabel('output 3');grid on;

linkaxes([x1,x2,x3],'x');

%% ==== Wavelet Denoising 2nd =====
ecg_wavelet2 = preprocess1(ecg_ICA2);

figure
x1 = subplot(2,1,1);
plot(t,ecg_wavelet2);xlabel('ECG2 wavelet');title("ICA2"); grid on;
ylim([-15 15]);

x3 = subplot(3,1,3);
plot(t,chieu2);xlabel('source ECG');grid on;
linkaxes([x1,x2,x3],'x');

correlation = corr(chieu2', ecg_wavelet2')


%do dinh R
[pamp1,ptime1]=pan_tompkin(-ecg_wavelet2,Fs,1);
%do dinh R
figure
[pamp2,ptime2]=pan_tompkin(-chieu2,Fs,1);



% corr_Tan  0.6585  0.6639
% 0.7502 0.6637  0.4224

% raw_ecg = 70 beats, qrs_ls = 140; qrsDuration = 0.0354; qrstime_all =70
% qrstime_allCopy = 70

% Tan = 73 beats; qrs_ls = 146; qrsDuration = 0.0586; qrstime_all2 = 73
% qrstime_all2 = 73

% Su = 71 beats; qrs_ls = 142; qrsDuration = 0.0606; qrstime_all2 = 71
% qrstime_all2 = 71
% 0.7512; 0.6751; 0.4419
