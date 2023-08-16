clear all; close all; clc

%% ======== Load Data EEG MIT =============
load_data = -1*load("C:\Users\Ho Van Su\Downloads\7R3-8R4.csv");
load_data=load_data(100:end,:);
for i=1:23
load_data(:,i)=load_data(:,i+1);
end

% load_data = load('C:\Users\Ma Pham Nhut Tan\Downloads\BrainFlow-RAW_2022-12-23_13-37-31_0.csv');
% load_data=load('c4o1.csv');

flag='db3';
% load('D:/EEG_Signal_Processing/DataBase/EEG_MIT/slp02am.mat');

Fs = 250;                                   % Sample Freq [Hz]
Ts = 1/Fs;                                  % Sample Rate [second]
t1 = 8;
range=20;% start [second]
t2 = 58;
% ch = 20;
% end [second]
% N = 160*(t2-t1);                            % Number of samples
t = t1:1/Fs:t2-Ts;
N = length(t);
f_axis = (0:N/2-1)*Fs/N;
% i=2;
%% Norm and mix
eeg1_bd = load_data(:,7)';
eeg2_bd = load_data(:,8)';
chieu_bd=load_data(:,2)';

eeg1_mix = eeg1(:,t1*Fs+1:(t2)*Fs);
eeg2_mix = eeg2(:,t1*Fs+1:(t2)*Fs);

eeg1=filterBPF_bd(eeg1_bd,Fs,1,100,8,1,"bandpass");
eeg1=filterBPF_bd(eeg1,Fs,40,65,8,1,"stop");
eeg1=filterBPF_bd(eeg1,Fs,95,105,8,1,"stop");

eeg2=filterBPF_bd(eeg2_bd,Fs,1,100,8,1,"bandpass");
eeg2=filterBPF_bd(eeg2,Fs,40,65,8,1,"stop");
eeg2=filterBPF_bd(eeg2,Fs,95,105,8,1,"stop");

chieu=filterBPF_bd(chieu_bd,Fs,1,100,8,1,"bandpass");
chieu=filterBPF_bd(chieu,Fs,40,65,8,1,"stop");
chieu=filterBPF_bd(chieu,Fs,95,105,8,1,"stop");

ecg = chieu_bd;
data = [eeg1; eeg2];
data_seg = data(:,t1*Fs+1:(t2)*Fs);
figure
subplot(2,1,1)
plot(t,eeg1(t1*Fs+1:t2*Fs));title('EEG 1');
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,eeg2(t1*Fs+1:t2*Fs));title('EEG 2');
xlim([t1 t1+range]);
%% ======== ICA1 : ECG Main & EEG Removed ECG =============
% data_ICA0 = ICA(data_MIT);
% eeg = data_ICA0(2,:);                
% ecg = data_ICA0(1,:);                %ECG main
% eeg = eeg*sqrt(var(eeg_MIT)/var(eeg)); 
% ecg = ecg*sqrt(var(ecg_MIT)/var(ecg)); 
% data_new = [eeg;ecg];
% 
% %% ========= Plot Data After ICA1  ===========
% figure
% s1 = subplot(2,1,1);
% plot(t,eeg(1:N), 'k'); grid on;
% % title('EEG after ICA 1st');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.025 0.045]);
% exportgraphics(s1, 's1.pdf', 'ContentType','vector');
% 
% s2 = subplot(2,1,2);
% plot(t,ecg(1:N), 'k'); grid on;
% % title('ECG after ICA 1st'); 
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.5 1]);
% exportgraphics(s2, 's2.pdf', 'ContentType','vector');
% 
% %% ======== Norm ecg and generate noises =============
% ecg_norm = ecg*sqrt(var(eeg)/var(ecg));            %ECG Normalized 
% n1 = sqrt(var(eeg))*rand(1,length(eeg));           %Noise 1 
% n2 = sqrt(var(eeg))*rand(1,length(eeg));           %Noise 2
% 
% figure
% noise1 = subplot(2,1,1);
% plot(t,n1(1:N), 'k'); grid on;
% % title('EEG after ICA 1st');
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
% 
% 
% %% ==========  Mix Data  ============
% a = 1./db2pow(24);                                      %EEG: 0db, ECG: -14 vs -12db, noise at -20db 
% mix1 = eeg + sqrt(0.04).*ecg_norm   + sqrt(a).*n1;      %EEG(1) + 0.2*EEG_norm(1) + 0.1*n1(1)
% mix2 = eeg + sqrt(0.0625).*ecg_norm + sqrt(a).*n2;
% data_mix1 = [mix1;mix2];
% 
% %% ========= Plot Data Mixed ===========
% figure
% x1 = subplot(2,1,1);
% plot(t, mix1(:,1:N), 'k');grid on;
% % title('The first output of mixer');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.025 0.045]);
% exportgraphics(x1, 'x1.pdf', 'ContentType','vector');
% 
% x2 = subplot(2,1,2);
% plot(t, mix2(:,1:N), 'k');grid on;
% % title('The second output of mixer');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.025 0.045]);
% exportgraphics(x2, 'x2.pdf', 'ContentType','vector');

% xlim([0 5]);
% ylim([-0.025 0.045]);
% figure
% plot(t, eeg(:,1:N));grid on;hold on
% plot(t, sqrt(a).*n1(:,1:N));
% plot(t, sqrt(0.04).*ecg_norm(:,1:N));
% plot(t, mix1(:, 1:N));
% xlim([0 5]);

%% ========== ICA2 ============
data_ICA1 = ICA(data_seg);
figure
subplot(2,1,1)
plot(t,data_ICA1(1,1:N));title('After ICA');
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,data_ICA1(2,1:N));
xlim([t1 t1+range]);
%% ========= Norm ICA Data ==========
eeg1_ICA = data_ICA1(1,:)*sqrt(var(eeg1)/var(data_ICA1(1,:))) ;
% ecg2_ICA = data_ICA1(2,:)*sqrt(var(eeg1)/var(data_ICA1(2,:))) ;
%% ======== Filter Bandpass ===========
% fc1 = 0.25; fc2 = 40; order = 4; mode = "Bandpass";
% ecg_ICA = double(ecg_ICA);
% [b,a] = butter(order, [fc1/(Fs/2) fc2/(Fs/2)],mode);
% h = fvtool(b,a);
% ecg_ICA = filtfilt(b,a,ecg_ICA);
% tmp_ECG = ecg_ICA;
%% ==== Smoooth EEG by Moving Average Filter =====
array = ones(1,8)/8;
eeg1_ICA = conv(data_ICA1(1,:), array, 'same');

figure
subplot(2,1,1)
plot(t,eeg1_ICA(:,1:N));title('After ICA');
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,data_ICA1(2,1:N));
xlim([t1 t1+range]);


% corre = corr(eeg1', eeg1_ICA')
%% ========= Correlation of Data after ICA 2nd =============
% [rownum1,colnum1] = size(data_ICA1);
% 
% fix_ICA = [];
% for i = 1:rownum1
%     h = data_new(i,:);
%     l = data_ICA1(i,:);
%     k = corr(h',l');
%     if (k < 0)
%         fprintf('Inverse Channel %d \n',i);
%         l = (-1)*l;
%         k = corr(h',l');
%     else  fprintf('Not Inverse Channel %d \n',i);
%     end
%     fprintf('Correlation of Channel %d after ICA :  %f  \n',i,k);
%     fix_ICA(i,:) = l;
% end

%% ========= Plot Data ICA2 after Norm & Inverse ===========
% figure
% s1_after_Normal_Smooth = subplot(2,1,1);
% plot(t, fix_ICA(1,1:N), 'k'); grid on
% % title('EEG after Normalizing & Smoothing');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.025 0.045]);
% exportgraphics(s1_after_Normal_Smooth, 's1_after_Normal_Smooth.pdf', 'ContentType','vector');
% 
% s2_after_Normal = subplot(2,1,2);
% plot(t, fix_ICA(2,1:N), 'k'); grid on
% % title('(ECG & Noises)'' after Normalizing');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude  [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% ylim([-0.5 1]);
% exportgraphics(s2_after_Normal, 's2_after_Normal.pdf', 'ContentType','vector');
%% === Diff of 2 signals ===
diff1 = eeg1_mix - eeg1_ICA;            %EEG + 0.2*ECG_norm + 0.1*n1 - êeg(smoothed)
diff2 = eeg2_mix - eeg1_ICA;        
data_mix2 = [diff1; diff2; data_ICA1(2,:)];


figure
subplot(3,1,1)
plot(t,eeg1_mix(1,t1*Fs+1:t2*Fs));title('EEG mix1'); hold on; grid on;
xlim([t1 t1+range]);
subplot(3,1,2)
plot(t,eeg2_mix(2,t1*Fs+1:t2*Fs));title('EEG mix2');
xlim([t1 t1+range]);
subplot(3,1,3)
plot(t,data_ICA1(2,t1*Fs+1:t2*Fs));title('ECG');
xlim([t1 t1+range]);


figure
subplot(3,1,1)
plot(t,eeg1_mix(:,1:N));title('After Diff'); hold on; grid on;
xlim([t1 t1+range]);
subplot(3,1,2)
plot(t,eeg2_mix(:,1:N));
xlim([t1 t1+range]);
subplot(3,1,3)
plot(t,eeg1_ICA(:,1:N));
xlim([t1 t1+range]);

% figure
% subplot(2,1,1);
% plot(t, diff1(1:N));grid on
% title('Diff 1');
% xlim([0 5]);
% subplot(2,1,2);
% plot(t, diff2(1:N));grid on
% title('Diff 2');
% xlim([0 5]);

%% ========== ICA 3rd ============
data_ICA2 = ICA(data_mix2);

% figure
% ECG_after_ICA2 = subplot(2,1,1);
% plot(t,-data_ICA2(1,1:N), 'k');grid on
% % title('ECG after ICA 3rd');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% exportgraphics(ECG_after_ICA2, 'ECG_after_ICA2.pdf', 'ContentType','vector');
% 
% subplot(3,1,2);
% plot(t,data_ICA2(2,1:N), 'k');grid on
% % title('Noises1'' after ICA 3rd');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);
% 
% subplot(3,1,3);
% plot(t,data_ICA2(3,1:N), 'k');grid on
% % title('Noises2'' after ICA 3rd');
% xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
% xlim([0 5]);

figure
subplot(3,1,1)
plot(t,data_ICA2(1,t1*Fs+1:t2*Fs));title('EEG mix1'); hold on; grid on;
xlim([t1 t1+range]);
subplot(3,1,2)
plot(t,data_ICA2(2,t1*Fs+1:t2*Fs));title('EEG mix2');
xlim([t1 t1+range]);
subplot(3,1,3)
plot(t,data_ICA2(3,t1*Fs+1:t2*Fs));title('ECG');
xlim([t1 t1+range]);


figure
subplot(3,1,1)
plot(t,data_ICA2(1,1:N));title('After ICA 2'); hold on; grid on;
xlim([t1 t1+range]);
subplot(3,1,2)
plot(t,data_ICA2(2,1:N));
xlim([t1 t1+range]);
subplot(3,1,3)
plot(t,data_ICA2(3,1:N));
xlim([t1 t1+range]);





tmp_ICA = -data_ICA2(1,:);
tmp_ICA = tmp_ICA*sqrt(var(ecg)/var(tmp_ICA));
tmp_ICA = double(tmp_ICA);

figure
ecg_after_ICA2_Normal = subplot(2,1,1);
plot(t,tmp_ICA(1:N), 'k');grid on
% title('ECG after ICA');
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);
exportgraphics(ecg_after_ICA2_Normal, 'ecg_after_ICA2_Normal.pdf', 'ContentType','vector');
%% ========== FFT ===============
[f0, P0] = fft_signal(ecg, Fs);
figure
plot(f0,P0) 
title("Single-Sided Amplitude Spectrum of Original ECG")
xlabel("f (Hz)")
ylabel("|P1(f)|")

[f, P] = fft_signal(tmp_ICA, Fs);
figure
plot(f,P) 
title("Single-Sided Amplitude Spectrum of ECG ICA3 Before Filtering")
xlabel("f (Hz)")
ylabel("|P1(f)|")

% ecg_bandpass = filterBPF(tmp_ICA,Fs,0.25,40,4,1,"Bandpass");
%% ======== Filter Bandpass ===========
fc1 = 0.25; fc2 = 40; order = 4; mode = "Bandpass";
[b,a] = butter(order, [fc1/(Fs/2) fc2/(Fs/2)],mode);
h = fvtool(b,a);
ecg_bandpass = filtfilt(b,a,tmp_ICA);
[f1, P1] = fft_signal(ecg_bandpass, Fs);
figure
plot(f1,P1) 
title("Single-Sided Amplitude Spectrum of Filtered ECG")
xlabel("f (Hz)")
ylabel("|P1(f)|")

corr_ecg = corr(ecg_bandpass' , ecg')

csvwrite("ecg_bandpass.csv", ecg_bandpass);
csvwrite("tmp.csv", tmp_ICA);
csvwrite("ecg.csv",ecg);

figure
subplot(2,1,1);
plot(t,tmp_ICA(1:N), 'k');grid on
% title('ECG after ICA');
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);

ecg_after_Filter = subplot(2,1,2);
plot(t,ecg_bandpass(1:N), 'k');grid on
% title('ECG after ICA');
xlabel('Time [s]','Interpreter','latex',"FontSize",13);ylabel('Amplitude [mV]','Interpreter','latex',"FontSize",13);
xlim([0 5]);
ylim([-0.5 1]);
exportgraphics(ecg_after_Filter, 'ecg_after_Filter.pdf', 'ContentType','vector');

%% ======= SNR without using methodology =======
% SNR_eeg = snr(eeg, eeg_ICA);
% fprintf('SNR EEG without using method: %.13f \n', SNR_eeg);
% SNR_ecg = snr(ecg, fix_ICA(2,:));
% fprintf('SNR ECG without using method: %.13f \n', SNR_ecg);

%% ========= SNR of Signal =============
% SNR_eeg1 = snr(eeg, ecg_ICA);
% fprintf('SNR EEG using method: %.13f \n', SNR_eeg1);
% SNR_ecg1 = snr(ecg, tmp_ICA);
% fprintf('SNR ECG using method: %.13f \n', SNR_ecg1);

%% ======== Without using method ======
% error1 = 0;
% for n = 1:length(eeg)
%     error  = abs(eeg_ICA1(n) - eeg(n));
%     error  = error1 + error;
%     error1 = error;
% end
% error = error/(colnum1); 
% fprintf('MAE EEG without using method: %.13f \n', error);
% 
% error2 = 0;
% for n = 1:length(ecg)
%     error_1 = abs(fix_ICA(2,n) - ecg(n));
%     error_1 = error2 + error_1;
%     error2 = error_1;
% end
% error_1 = error_1/(colnum1); 
% fprintf('MAE ECG without using method: %.13f \n', error_1);
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
fprintf('SNR of ECG using Ordinary ICA: %.13f \n', SNR);

error2 = 0;
for n = 1:length(ecg)
    error_method  = (ecg(n) - tmp_ICA(n)).^2;
    error_method  = error_method + error2;
    error2 = error_method;
end
SNR1 = 10*log(s/error2); 
fprintf('SNR of ECG using Proposed technique: %.13f \n', SNR1);


%% ======== Filter Bandpass ===========
fc1 = 0.25; fc2 = 40; order = 4; mode = "Bandpass";
tmp_ECG = double(tmp_ECG);
[b,a] = butter(order, [fc1/(Fs/2) fc2/(Fs/2)],mode);
h = fvtool(b,a);
tmp_corr = filtfilt(b,a,tmp_ECG);

corr_ecg_ICA2 = corr(ecg', -data_ICA2(1,:)')

corr_ECG1 = corr(ecg', tmp_corr')

corr_ECG2 = corr_ecg