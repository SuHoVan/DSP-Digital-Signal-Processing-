clear all
close all;clc

load_data = -1*load("C:\Users\Ho Van Su\Downloads\7X1_8X2_final.csv");
load_data=load_data(100:end,:);
for i=1:23
load_data(:,i)=load_data(:,i+1);
end

flag='db3';

Fs = 250;                                   % Sample Freq [Hz]
Ts = 1/Fs;                                  % Sample Rate [second]
t1 = 13;
range=60;% start [second]
t2 = 73;
t = t1:1/Fs:t2-Ts;
N = length(t);
f_axis = (0:N/2-1)*Fs/N;

%% Norm and mix
eeg1 = load_data(:,7)';
eeg2 = load_data(:,8)';
chieu=load_data(:,1)';
chieu2=load_data(:,2)';
chieu3=load_data(:,3)';

figure
subplot(2,1,1)
plot(t,eeg1(t1*Fs+1:t2*Fs));xlabel('EEG 1');title("Raw signal")
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,eeg2(t1*Fs+1:t2*Fs));xlabel('EEG 2');
xlim([t1 t1+range]);


eeg1=filterBPF(eeg1,Fs,1,100,8,1,"bandpass");
eeg1=filterBPF(eeg1,Fs,40,65,8,1,"stop");
eeg1=filterBPF(eeg1,Fs,95,105,8,1,"stop");
eeg2=filterBPF(eeg2,Fs,1,100,8,1,"bandpass");
eeg2=filterBPF(eeg2,Fs,40,65,8,1,"stop");
eeg2=filterBPF(eeg2,Fs,95,105,8,1,"stop");
chieu=filterBPF(chieu,Fs,1,100,8,1,"bandpass");
chieu=filterBPF(chieu,Fs,40,65,8,1,"stop");
chieu=filterBPF(chieu,Fs,95,105,8,1,"stop");
chieu2=filterBPF(chieu2,Fs,1,100,8,1,"bandpass");
chieu2=filterBPF(chieu2,Fs,40,65,8,1,"stop");
chieu2=filterBPF(chieu2,Fs,95,105,8,1,"stop");
chieu3=filterBPF(chieu3,Fs,1,100,8,1,"bandpass");
chieu3=filterBPF(chieu3,Fs,40,65,8,1,"stop");
chieu3=filterBPF(chieu3,Fs,95,105,8,1,"stop");
%---------------------
figure
subplot(2,1,1)
plot(t,eeg1(t1*Fs+1:t2*Fs));xlabel('EEG 1');title("After filter")
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,eeg2(t1*Fs+1:t2*Fs));xlabel('EEG 2');
xlim([t1 t1+range]);

%-----------------
helo1=eeg1;
helo2=eeg2;
datahelo=[helo1 helo2];
data = [eeg1;eeg2];

%--------------------
data_seg = data(:,t1*Fs+1:(t2)*Fs);
data_ICA = ICA_Kur(data_seg,Fs);


figure
subplot(2,1,1)
plot(t,data_ICA(1,1:N));title('After ICA');
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,data_ICA(2,1:N));
xlim([t1 t1+range]);

%%
%fft
fftin=eeg2(1:60*Fs);
figure
f_axis = (0:length(fftin)/2-1)*Fs/(length(fftin));
time = linspace(t1,t2,length(fftin));
fft1 = fft(fftin);
fft1 = abs(fft1(1:length(fftin)/2))/(length(fftin)/2);
plot(f_axis,fft1); grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude')
title('before ICA')


[c,l] = wavedec(data_ICA(2,1:N),5,flag);
approx = appcoef(c,l,flag);
[cd1,cd2,cd3,cd4,cd5] = detcoef(c,l,[1 2 3 4 5]);
f_axis1 = (0:l(6)/2-1)*Fs/(2^1*l(6));
time1 = linspace(t1,t2,l(6));
f_axis2 = (0:l(5)/2-1)*Fs/(2^2*l(5));
time2 = linspace(t1,t2,l(5));
f_axis3 = (0:l(4)/2-1)*Fs/(2^3*l(4));
time3 = linspace(t1,t2,l(4));
f_axis4 = (0:l(3)/2-1)*Fs/(2^4*l(3));
time4 = linspace(t1,t2,l(3));
f_axis5 = (0:l(2)/2-1)*Fs/(2^5*l(2));
time5 = linspace(t1,t2,l(2));
figure
subplot(3,2,1)
fft_app = fft(approx);
fft_app = abs(fft_app(1:l(1)/2))/(l(1)/2);
plot(time5,approx);
xlim([t1 t1+range]);
title('Approximation Coefficients')
%------------------------
subplot(3,2,2)
fft_cd5 = fft(cd5);
fft_cd5 = abs(fft_cd5(1:l(2)/2))/(l(2)/2);
plot(time5,cd5);
xlim([t1 t1+range]);
title('Level 5 Detail Coefficients')
%------------------------
subplot(3,2,3)
fft_cd4 = fft(cd4);
fft_cd4 = abs(fft_cd4(1:l(3)/2))/(l(3)/2);
plot(time4,cd4);
xlim([t1 t1+range]);
title('Level 4 Detail Coefficients')
%------------------------
subplot(3,2,4)
fft_cd3 = fft(cd3);
fft_cd3 = abs(fft_cd3(1:l(4)/2))/(l(4)/2);
plot(time3,cd3);
xlim([t1 t1+range]);
title('Level 3 Detail Coefficients')
%------------------------
subplot(3,2,5)
fft_cd2 = fft(cd2);
fft_cd2 = abs(fft_cd2(1:l(5)/2))/(l(5)/2);
plot(time2,cd2);
xlim([t1 t1+range]);
title('Level 2 Detail Coefficients')
%------------------------
subplot(3,2,6)
fft_cd1 = fft(cd1);
fft_cd1 = abs(fft_cd1(1:l(6)/2))/(l(6)/2);
plot(time1,cd1);
xlim([t1 t1+range]);
title('Level 1 Detail Coefficients')
%% ------Freq Domain
figure
subplot(3,2,1)
plot(f_axis5,fft_app); grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [\muV]')
title('Approximation Coefficients')
%------------------------
subplot(3,2,2)
plot(f_axis5,fft_cd5); grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [\muV]')
title('Level 5 Detail Coefficients')
%------------------------
subplot(3,2,3)
plot(f_axis4,fft_cd4); grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [\muV]')
title('Level 4 Detail Coefficients')
%------------------------
subplot(3,2,4)
plot(f_axis3,fft_cd3); grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [\muV]')
title('Level 3 Detail Coefficients')
%------------------------
subplot(3,2,5)
plot(f_axis2,fft_cd2); grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [\muV]')
title('Level 2 Detail Coefficients')
%------------------------
subplot(3,2,6)
plot(f_axis1,fft_cd1); grid on;
xlabel('Frequency [Hz]');
ylabel('Amplitude [\muV]')
title('Level 1 Detail Coefficients')

%%

dnh1=0.1745;
    dnh2=2.9745;
    dnh3=15.9745;
    dnh4=13.9745;
    dnh5=11.1755;
    dnh6=0.1755;
    dnh7=0.1755;


[lamda1 delta1] = Compute_lamda(cd1,dnh1);
figure
subplot(2,1,1)
plot(time1,cd1);
title('Level 1 Detail Coefficients')
xlim([0 5]);
subplot(2,1,2)
plot(time1,delta1);
xlim([0 5]);

[lamda2 delta2] = Compute_lamda(cd2,dnh2);
figure
subplot(2,1,1)
plot(time2,cd2);
title('Level 2 Detail Coefficients')
xlim([0 5]);
subplot(2,1,2)
plot(time2,delta2);
xlim([0 5]);

[lamda3 delta3] = Compute_lamda(cd3,dnh3);
figure
subplot(2,1,1)
plot(time3,cd3);
title('Level 3 Detail Coefficients')
xlim([0 5]);
subplot(2,1,2)
plot(time3,delta3);
xlim([0 5]);

[lamda4 delta4] = Compute_lamda(cd4,dnh4);
figure
subplot(2,1,1)
plot(time4,cd4);
title('Level 4 Detail Coefficients')
xlim([0 5]);
subplot(2,1,2)
plot(time4,delta4);
xlim([0 5]);

[lamda5 delta5] = Compute_lamda(cd5,dnh5);
figure
subplot(2,1,1)
plot(time5,cd5);
title('Level 5 Detail Coefficients')
xlim([0 5]);
subplot(2,1,2)
plot(time5,delta5);
xlim([0 5]);

helooo=data_ICA';
%%
figure
cmix= [approx delta5 delta4 delta3 delta2 delta1];
cmix=filt(waverec(cmix,l,flag),0.5,100,250);
subplot(2,1,1)
plot(t,cmix);
title('reconstruct');
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,eeg2(t1*Fs+1:t2*Fs));title('EEG');
xlim([t1 t1+range]);

figure
subplot(2,1,1)
plot(t,cmix);
title('reconstruct');
xlim([t1 t1+range]);
subplot(2,1,2)
plot(t,chieu(t1*Fs+1:t2*Fs)');title('ECG');
xlim([t1 t1+range]);

%correlation
chieu1=chieu(t1*Fs+1:t2*Fs);
chieu2=chieu2(t1*Fs+1:t2*Fs);
chieu3=chieu3(t1*Fs+1:t2*Fs);
co1=corr(cmix',chieu1')
co2=corr(cmix',chieu2')
co3=corr(cmix',chieu3')

%do dinh R
%do dinh R
figure
[pamp1,ptime1]=pan_tompkin(chieu1,Fs,1);
figure
[pamp2,ptime2]=pan_tompkin(-cmix,Fs,1);
%do dinh R
figure
[pamp3,ptime3]=pan_tompkin(-chieu2,Fs,1);
%do dinh R
figure
[pamp4,ptime4]=pan_tompkin(chieu3,Fs,1);

%khoang QRS
% ECG original
% [qrs_ls,qrs] = calc_qrs(chieu1, ptime1);
% qrstime_all=qrs./Fs;
% qrsDuration=mean(qrstime_all);
% hello = chieu1;
% figure
% plot(1:length(hello), hello, qrs_ls, hello(qrs_ls), 'r*')
% % % ECG tach
% [qrs_ls_2,qrs2] = calc_qrs(cmix, ptime2);
% qrstime_all_2=qrs2./Fs;
% qrsDuration2=mean(qrstime_all_2);
% hello2 = cmix;
% figure
% plot(1:length(hello2), hello2, qrs_ls_2, hello2(qrs_ls_2), 'r*')


% Fs = 250;                                   % Sample Freq [Hz]
% Ts = 1/Fs;                                  % Sample Rate [second]
% t1 = 0;
% t2 = 50;
% t = t1:1/Fs:t2-Ts;
% a = ecg-mean(ecg)
% figure
% subplot(2,1,1)
% plot(t,a);grid on; hold on;
% axis tight;
%    line(repmat(qrs_i_raw,[2 1]),...
%        repmat([min(ecg-mean(ecg))/2; max(ecg-mean(ecg))/2],size(qrs_i_raw)),...
%        'LineWidth',2,'LineStyle','-.','Color','r');
%     zoom on;
