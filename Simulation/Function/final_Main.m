clear all; close all; clc

%% ======== Load Data EEG MIT =============
load_data_MIT = load("D:\EEG\data.csv", '-ascii');
eeg = load_data_MIT(1,:);
ecg = load_data_MIT(2,:);
    
% load_data_MIT = load("D:\EEG\Source\2a_scaled.csv", '-ascii')';
% load_data_MIT = load("D:\EEG\Source\3_scaled.csv", '-ascii')';
% load_data_MIT = load("D:\EEG\Source\16_scaled.csv", '-ascii')';
% eeg = load_data_MIT(3,:);
% ecg = load_data_MIT(1,:);


ma = load("D:\EEG\File data\mam.mat");
em = load("D:\EEG\File data\emm.mat");

ma = ma.val/1000;
em = em.val/1000;

ma = ma(1,:);
em = em(1,:);

noise1 = 0.85.*ma + 0.7.*em;
noise2 = 0.8.*ma + 0.9.*em;
% Define the original sampling rate
original_sampling_rate = 360;
% Define the desired sampling rate
desired_sampling_rate = 250;
% Calculate the resampling ratio
resampling_ratio = desired_sampling_rate / original_sampling_rate;
% Resample the signal
noise1 = resample(noise1, desired_sampling_rate, original_sampling_rate);
noise2 = resample(noise2, desired_sampling_rate, original_sampling_rate);

Fs = 250;                                   % Sample Freq [Hz]
Ts = 1/Fs;                                  % Sample Rate [second]
t1 = 0; t2 = 180;
x = t1:1/Fs:t2-Ts;
N = length(x);

eeg = eeg(:,t1*Fs+1:(t2)*Fs);
ecg = ecg(:,t1*Fs+1:(t2)*Fs);
noise1 = noise1(:,t1*Fs+1:(t2)*Fs);
noise2 = noise2(:,t1*Fs+1:(t2)*Fs);
%% ======== Norm ecg and generate noises =============
ecg_norm = ecg*sqrt(var(eeg)/var(ecg));            %ECG Normalized 
noise_gauss1 = (-0.045+rand(1,N)*(0.045-(-0.045)));
noise_gauss2 = (-0.045+rand(1,N)*(0.045-(-0.045)));
n1 = (noise_gauss1 + noise1);
n1 = n1.*sqrt(var(eeg)/var(n1));
n2 = (noise_gauss2 + noise2);
n2 = n2.*sqrt(var(eeg)/var(n2));
%% ==========  Mix Data  ============
a = 1./db2pow(20);                                      %EEG: 0db, ECG: -14 vs -12db, noise at -20db 
mix1 = eeg + sqrt(0.04).*ecg_norm   + sqrt(a).*n1;      %EEG(1) + 0.2*EEG_norm(1) + 0.1*n1(1)
mix2 = eeg + sqrt(0.0625).*ecg_norm + sqrt(a).*n2;
data_mix1 = [mix1;mix2];

%% ========= Test power of signals =========== 
% k = var(eeg)
% b = var(sqrt(0.04).*ecg_norm)
% c = var(sqrt(a).*n1)
% pow2db(b/k)
% pow2db(c/k)

%% ========= Plot Data  ===========
figure
s1 = subplot(2,1,1);
plot(x,eeg(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on;
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.04]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

s2 = subplot(2,1,2);
plot(x,ecg_norm(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on; 
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
% ylim([-0.5 1]);
linkaxes([s1,s2],'x');
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

figure
noise1 = subplot(2,1,1);
plot(x,n1(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on;
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.04]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

noise2 = subplot(2,1,2);
plot(x,n2(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on;
% title('ECG after ICA 1st'); 
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.04]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

linkaxes([noise1,noise2],'x');

%% ========= Plot Data Mixed ===========
figure
x1 = subplot(2,1,1);
plot(x, mix1(:,1:N), 'Color' , '#0072BD','LineWidth',1.2);grid on;
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.04]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

x2 = subplot(2,1,2);
plot(x, mix2(:,1:N),'Color' , '#0072BD','LineWidth',1.2);grid on;
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.04]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

linkaxes([x1,x2],'x');



%% ========== ICA 1st ============
data_ICA1 = ICA(data_mix1);

% ========= Output ICA 1st ==========
eeg_ICA = -data_ICA1(1,:);
ecg_ICA = data_ICA1(2,:);


% corr(ecg', test')
figure;
x3 = subplot(2,1,1);
plot(x, -data_ICA1(1,1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.04]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

x4 = subplot(2,1,2);
plot(x, data_ICA1(2,1:N),'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.015 0.03]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);
linkaxes([s1,s2],'x');



ecg_ICA = remove_dt(t,ecg_ICA);
test_ecg = ecg_ICA;


%% ==== Wavelet Denoising (EEG^) =====
eeg_wavelet = preprocess(eeg_ICA);

figure;
subplot(2,1,1);
plot(x, eeg_wavelet(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.04]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

subplot(2,1,2);
plot(x, ecg_ICA(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.02 0.03]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);


%% === SIC process ===
diff1 = mix1 - eeg_wavelet;            %EEG + 0.2*ECG_norm + 0.1*n1 - êeg(smoothed)
diff2 = mix2 - eeg_wavelet;

figure;
subplot(2,1,1);
plot(x, diff1(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.015 0.03]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

subplot(2,1,2);
plot(x, diff2(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.015 0.03]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);
data_mix2 = [diff1; diff2; ecg_ICA];

%% ========== ICA 2nd ============
data_ICA2 = ICA(data_mix2);

ecg_ICA2 = -data_ICA2(1,:);
ecg_ICA2 = remove_dt(t,ecg_ICA2);


figure;
subplot(2,1,1);
plot(x, -data_ICA2(1,1:N), 'Color' , '#0072BD','LineWidth',1.15); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.015 0.03]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

subplot(2,1,2);
plot(x, -data_ICA2(2,1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);


figure;
subplot(2,1,1);
plot(x, -data_ICA2(3,1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);


%% ======== Wavelet Denosing 2nd ==========
ecg_wavelet = preprocess1(ecg_ICA2);
corr_ecg = corr(ecg_wavelet' , ecg');

%% ========== Plot ECG ===========

% figure
% x1 = subplot(5,1,1);
% plot(x,ecg_ICA(1:N), 'k','LineWidth',1.15);grid on
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.015 0.03]);
% 
% x2 = subplot(5,1,2);
% plot(x,test(1:N), 'k','LineWidth',1.15);grid on
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.015 0.03]);
% 
% x3 = subplot(5,1,3);
% plot(x,ecg_ICA2(1:N), 'k','LineWidth',1.15);grid on
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.015 0.03]);
% 
% x4 = subplot(5,1,4);
% plot(x,ecg_wavelet(1:N), 'k','LineWidth',1.15);grid on
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.015 0.03]);
% 
% 
% x5 = subplot(5,1,5);
% plot(x,ecg_norm(1:N), 'k','LineWidth',1.15);grid on
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.05 0.1]);
% 
% 
% linkaxes([x1,x2,x3,x4,x5], 'x');


figure;
subplot(2,1,1);
plot(x, test(1:N),'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.015 0.03]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

subplot(2,1,2);
plot(x, ecg_wavelet(1:N), 'Color' , '#0072BD','LineWidth',1.2); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.015 0.03]);
pax.GridAlpha = 0.2;
set(gca, 'LineWidth', 1.2);

figure
[pamp2,ptime2]=pan_tompkin(ecg_wavelet,Fs,1);
figure
[pamp3,ptime3]=pan_tompkin_ecg(ecg,Fs,1);