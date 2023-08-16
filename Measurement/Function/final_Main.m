clear all; close all; clc

%% ======== Load Data EEG MIT =============
load_data_MIT = load("D:\EEG\data.csv", '-ascii')';
eeg = load_data_MIT(:,1)';
ecg = load_data_MIT(:,2)';
data = [eeg;ecg];

Fs = 250;                                   % Sample Freq [Hz]
Ts = 1/Fs;                                  % Sample Rate [second]
t1 = 0; t2 = 120;
x = t1:1/Fs:t2-Ts;
N = length(x);

% [eeg] = remove_dt(t, eeg_MIT);
% [ecg] = remove_dt(t, ecg_MIT);
% data_new = [eeg;ecg];
% 
% data_new = ICA(data_new);
% eeg = data_new(2,:); 
% ecg = data_new(1,:); 
% [eeg] = remove_dt(t, eeg);
% [ecg] = remove_dt(t, ecg);
% eeg = eeg*sqrt(var(eeg_MIT)/var(eeg));
% ecg = ecg*sqrt(var(ecg_MIT)/var(ecg));
% data = [eeg;ecg];
%% ========= Plot Data  ===========
figure
s1 = subplot(2,1,1);
plot(x,eeg(1:N), 'k'); grid on;
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.045]);

s2 = subplot(2,1,2);
plot(x,ecg(1:N), 'k'); grid on; 
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);
linkaxes([s1,s2],'x');
%% ======== Norm ecg and generate noises =============
ecg_norm = ecg*sqrt(var(eeg)/var(ecg));            %ECG Normalized 
n1 = sqrt(var(eeg))*rand(1,length(eeg));           %Noise 1 
n2 = sqrt(var(eeg))*rand(1,length(eeg));           %Noise 2

% figure
% noise1 = subplot(2,1,1);
% plot(t,n1(1:N), 'k'); grid on;
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.003 0.015]);
% exportgraphics(noise1, 'n1.pdf', 'ContentType','vector');
% 
% noise2 = subplot(2,1,2);
% plot(t,n2(1:N), 'k'); grid on;
% % title('ECG after ICA 1st'); 
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.003 0.015]);
% exportgraphics(noise2, 'n2.pdf', 'ContentType','vector');


%% ==========  Mix Data  ============
a = 1./db2pow(22);                                      %EEG: 0db, ECG: -14 vs -12db, noise at -20db 
mix1 = eeg + sqrt(0.04).*ecg_norm   + sqrt(a).*n1;      %EEG(1) + 0.2*EEG_norm(1) + 0.1*n1(1)
mix2 = eeg + sqrt(0.0625).*ecg_norm + sqrt(a).*n2;
data_mix1 = [mix1;mix2];


data_mix1 = data_mix1(:,t1*Fs+1:(t2)*Fs);
mix1 = mix1(:,t1*Fs+1:(t2)*Fs);
mix2 = mix2(:,t1*Fs+1:(t2)*Fs);
eeg = eeg(:,t1*Fs+1:(t2)*Fs);
ecg = ecg(:,t1*Fs+1:(t2)*Fs);

mix_test1 = sqrt(0.04).*ecg_norm   + sqrt(a).*n1;
mix_test2 = sqrt(0.0625).*ecg_norm   + sqrt(a).*n2;
%% ========= Plot Data Mixed ===========
figure
x1 = subplot(2,1,1);
plot(x, mix1(:,1:N), 'k');grid on;
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.03 0.045]);


x2 = subplot(2,1,2);
plot(x, mix2(:,1:N), 'k');grid on;
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.03 0.045]);
linkaxes([x1,x2],'x');

%% ========== ICA2 ============
data_ICA1 = ICA(data_mix1);
corre = corr(eeg', -data_ICA1(1,:)');
t = 1:length(eeg);
test = data_ICA1(2,:);
test = remove_dt(t,test);
test = test*sqrt(var(ecg)/var(test));
test = preprocess(test);

corr(ecg', test')


figure;
x3 = subplot(2,1,1);
plot(x, -data_ICA1(1,1:N), 'k'); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.045]);

x4 = subplot(2,1,2);
plot(x, data_ICA1(2,1:N), 'k'); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.015 0.03]);
linkaxes([s1,s2],'x');


% ========= Normalize ICA Data ==========
eeg_ICA = -data_ICA1(1,:);
ecg_ICA = data_ICA1(2,:);
t = 1:length(eeg);
%% ==== Smoooth EEG by Moving Average Filter =====
eeg_ICA = remove_dt(t,eeg_ICA);
ecg_ICA = remove_dt(t,ecg_ICA);

eeg_ICA = eeg_ICA*sqrt(var(eeg)/var(eeg_ICA));
ecg_ICA = ecg_ICA*sqrt(var(ecg)/var(ecg_ICA));
test_ecg = ecg_ICA;

eeg_ICA = preprocess(eeg_ICA);

corr_eegICA1 = corr(eeg_ICA', eeg');
corr_ecgICA1 = corr(ecg_ICA', ecg');


figure;
subplot(2,1,1);
plot(x, eeg_ICA(1:N), 'k'); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.025 0.045]);

subplot(2,1,2);
plot(x, ecg_ICA(1:N), 'k'); grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);

%% === Diff of 2 signals ===
diff1 = mix1 - eeg_ICA;            %EEG + 0.2*ECG_norm + 0.1*n1 - êeg(smoothed)
diff2 = mix2 - eeg_ICA;

% figure
% ax1 = subplot(2,1,1);
% stem(x,mix1(1:N), 'k'); grid on;hold on;
% stem(x,eeg_ICA(1:N), 'r'); 
% stem(x,eeg(1:N), 'g'); 
% legend('mix1', 'eegICA','eeg');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% % xlim([0 5]);
% 
% ax2 = subplot(2,1,2);
% stem(x,mix_test1(1:N),'b');grid on; hold on;
% stem(x,ecg(1:N), 'g'); 
% stem(x,diff1(1:N),'p');
% legend('mix test1','ecg', 'diff1');
% % xlim([0 5]);
% linkaxes([ax1,ax2],'x');

% [f1, P1] = fft_signal(eeg, Fs);
% [f2, P2] = fft_signal(eeg_ICA, Fs);
% figure
% plot(f1,P1); hold on; plot(f2,P2) 

data_mix2 = [diff1; diff2; ecg_ICA];

%% ========== ICA 3rd ============
data_ICA2 = ICA(data_mix2);
ecg_ICA2 = -data_ICA2(1,:);
ecg_ICA2 = remove_dt(t,ecg_ICA2);
ecg_ICA2 = ecg_ICA2*sqrt(var(ecg)/var(ecg_ICA2));
corr_ecgICA2 = corr(ecg', ecg_ICA2');


%% ======== Filter Bandpass ===========
% fc1 = 0.25; fc2 = 40; order = 4; mode = "Bandpass";
% [b,a] = butter(order, [fc1/(Fs/2) fc2/(Fs/2)],mode);
% h = fvtool(b,a);

% ecg_bandpass = filtfilt(b,a,ecg_ICA2);
ecg_wavelet = preprocess(ecg_ICA2);
corr_ecg = corr(ecg_wavelet' , ecg');

fprintf('Corr EEG 1st ICA : %.4f \n', corre);
fprintf('Corr EEG wavelet denoise : %.4f \n', corr_eegICA1);
fprintf('Corr ECG 1st ICA: %.4f \n', corr_ecgICA1);
fprintf('Corr ECG 2nd: %.4f \n', corr_ecgICA2);
fprintf('Corr ecg_ICA wavelet denoise: %.4f \n', corr_ecg);
%% ========== Plot ECG ===========
figure
subplot(3,1,1);
plot(x,test(1:N), 'k');grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);

subplot(3,1,2);
plot(x,ecg_ICA2(1:N), 'k');grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);

subplot(3,1,3);
plot(x,ecg_wavelet(1:N), 'k');grid on
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);

%% ======== SNR ======
error1 = 0;
s1 = 0;
for n = 1:length(ecg)
    s = (ecg(n)).^2;
    s = s + s1;
    s1 = s;
    error  = (ecg(n) - ecg_ICA(n)).^2;
    error  = error + error1;
    error1 = error;
end
SNR = 10*log(s/error); 
fprintf('SNR of ECG using ICA one time: %.4f \n', SNR);

error2 = 0;
for n = 1:length(ecg)
    error_method  = (ecg(n) - ecg_ICA2(n)).^2;
    error_method  = error_method + error2;
    error2 = error_method;
end
SNR1 = 10*log(s/error_method); 
fprintf('SNR of ECG using method SIC: %.4f \n', SNR1);

