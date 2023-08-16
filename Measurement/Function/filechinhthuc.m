close all; clear all; clc

% %kênh7 não
fs =250;
% raw_data = readtable('D:\EEG\EEGafterICA.csv');
% raw_data = table2array(raw_data);
% channel7 = raw_data(:,1);

%kenh 8 não
raw_data1 = readtable("C:\Users\Ho Van Su\Downloads\7X1_8X2_final.csv");
raw_data1 = table2array(raw_data1);
channel7 = raw_data1(:,9)';
channel8 = raw_data1(:,8)';

fc1 = 1; fc2 = 100; order = 4; mode = "Bandpass";
[b,a] = butter(order, [fc1/(fs/2) fc2/(fs/2)],mode);
filtered_channel7 = filtfilt(b, a, channel7);
filtered_channel8 = filtfilt(b, a, channel8);

d1 = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
               'DesignMethod','butter','SampleRate',fs);

filtered_channel7 = filtfilt(d1,filtered_channel7);
filtered_channel8 = filtfilt(d1,filtered_channel8);
d2 = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',49,'HalfPowerFrequency2',51, ...
               'DesignMethod','butter','SampleRate',fs);

filtered_channel7 = filtfilt(d2,filtered_channel7);
filtered_channel8 = filtfilt(d2,filtered_channel8);


t = 1:length(filtered_channel7);
filtered_channel7 = remove_dt(t,filtered_channel7);
filtered_channel8 = remove_dt(t,filtered_channel8);

max_c7 = max(filtered_channel7(:));
min_c7 = min(filtered_channel7(:));
filtered_channel7   = (filtered_channel7 - min_c7) / (max_c7 - min_c7);

max_c8 = max(filtered_channel8(:));
min_c8 = min(filtered_channel8(:));
filtered_channel8   = (filtered_channel8 - min_c8) / (max_c8 - min_c8);

data = [filtered_channel7; filtered_channel8];

% Kênh 7
N = length(filtered_channel7);
% Y7 = fft(filtered_channel7, N)/N;
% f7 = fs/2*linspace(0, 1, N/2+1);
   
% figure;
% subplot(2, 1, 1);
% plot(f7, 2*abs(Y7(1:N/2+1)));
% xlabel('tan so HZ');
% ylabel('bien do');
% title('Kênh 7 - MiEN TAN SO');
 
% Kênh 8
% Y8 = fft(filtered_channel8, N)/N;
% f8 = fs/2*linspace(0, 1, N/2+1);

% subplot(2, 1, 2);
% plot(f8, 2*abs(Y8(1:N/2+1)));
% xlabel('tan so HZ');
% ylabel('bien do');
% title('Kênh 8 - MiEN TAN SO');
%  %ve tun hieu kenh 8

figure;
subplot(2,1,1);
plot(channel7);
xlabel('mienthoigian 7' );
ylabel('um/v');
title('tin hieu goc ');

subplot(2,1,2); 
plot(filtered_channel7);
xlabel('Time (s)');
ylabel('Voltage (uV)');
title('tin qua loc bandpass 7');

figure(1);
x1 = subplot(2,1,1);
plot(filtered_channel7);
x2 = subplot(2,1,2);
title('Source signal');
plot(filtered_channel8);
linkaxes([x1,x2],'x');



data_ICA = ICA(data);
eeg_ICA = data_ICA(2,:);
ecg_ICA = data_ICA(1,:);


max_eeg = max(eeg_ICA(:));
min_eeg = min(eeg_ICA(:));
eeg_ICA   = (eeg_ICA - min_eeg) / (max_eeg - min_eeg);

max_ecg = max(ecg_ICA(:));
min_ecg = min(ecg_ICA(:));
ecg_ICA   = (ecg_ICA - min_ecg) / (max_ecg - min_ecg);


figure(2);
x1 = subplot(2,1,1);
plot(eeg_ICA);
title('After ICA');
x2 = subplot(2,1,2); 
plot(ecg_ICA);
linkaxes([x1,x2],'x');


figure(3);
x1 = subplot(2,1,1);
plot(eeg_ICA,'r'); grid on; hold on;
plot(filtered_channel7,'k');

x2 = subplot(2,1,2); 
plot(eeg_ICA,'r');grid on; hold on;
plot(filtered_channel8,'k');
linkaxes([x1,x2],'x');

% eeg_norm = eeg_ICA*sqrt(var()/var(eeg_ICA));





% % Define frequency bands for filtering
% alpha_band = [8 12];
% beta_band = [13 30];
% delta_band = [1 4];
% gamma_band = [30 100];
% theta_band = [4 8]; % Add theta band
% 
% 
% 
% % Filter the signal using a bandpass filter for each frequency band
% alpha_filtered7 = bandpass(filtered_channel7, alpha_band, fs);
% beta_filtered7 = bandpass(filtered_channel7, beta_band, fs);
% delta_filtered7 = bandpass(filtered_channel7, delta_band, fs);
% gamma_filtered7 = bandpass(filtered_channel7, gamma_band, fs);
% theta_filtered7 = bandpass(filtered_channel7, theta_band, fs); % Add theta filtered signal
% %chanel8`   
% alpha_filtered = bandpass(filtered_channel8, alpha_band, fs);
% beta_filtered = bandpass(filtered_channel8, beta_band, fs);
% delta_filtered = bandpass(filtered_channel8, delta_band, fs);
% gamma_filtered = bandpass(filtered_channel8, gamma_band, fs);
% theta_filtered = bandpass(filtered_channel8, theta_band, fs); % Add theta filtered signal
% 
% 
% 
% % Time domain signal visualization kenh 8
% figure;
% t = (0:length(filtered_channel8)-1)/fs;
% subplot(2,3,2); 
% plot(t, alpha_filtered);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kênh 8 Alpha Wave - Time Domain');
% 
% 
% 
% subplot(2,3,3); 
% plot(t, beta_filtered);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kênh 8 Beta Wave - Time Domain');
% 
% subplot(2,3,4);
% plot(t, theta_filtered);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kênh 8 Theta Wave - Time Domain');
% 
% 
% subplot(2,3,5);
% plot(t, delta_filtered);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kênh 8 Delta Wave - Time Domain');
% 
% 
% subplot(2,3,6);
% plot(t, gamma_filtered);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kênh 8 Gamma Wave - Time Domain');
% 
% 
% ...
% 
% 
% %kênh8
% % Frequency domain signal visualization
% n = length(filtered_channel8);
% f = (0:n-1)*(fs/n);
% 
% alpha_fft = fft(alpha_filtered);
% beta_fft = fft(beta_filtered);
% delta_fft = fft(delta_filtered);
% gamma_fft = fft(gamma_filtered);
% theta_fft = fft(theta_filtered); % Add theta FFT
% 
% figure;
% subplot(2,3,1); % Change subplot to accommodate theta wave
% plot(f, abs(alpha_fft)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title(' kênh 8 Alpha Wave - Frequency Domain');
% xlim([0 30]);
% 
% 
% subplot(2,3,2); % Change subplot to accommodate theta wave
% plot(f, abs(beta_fft*1000));
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title(' kênh 8 Beta Wave - Frequency Domain');
% xlim([0 30]);
% 
% subplot(2,3,3); % Change subplot to accommodate theta wave
% plot(f, abs(theta_fft*1000));
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title('kênh 8 Theta Wave - Frequency Domain');
% xlim([0 8]); % Set x-axis limit to theta band
% 
% subplot(2,3,4); % Add theta wave subplot
% plot(f, abs(delta_fft)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title('kênh 8 Delta Wave - Frequency Domain');
% xlim([0 5]);
% 
% subplot(2,3,5);
% plot(f, abs(gamma_fft)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title('  kênh 8 Gamma Wave - Frequency Domain');
% xlim([0 100]);
% 
% 
% %chanel 7
% % Time domain signal visualization
% t1 = (0:length(filtered_channel7)-1)/fs;
% figure;
% subplot(2,3,2); % Change subplot to accommodate theta wave
% plot(t1, alpha_filtered7);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kenh thu 7 Alpha Wave - Time Domain');
% ylim ([-100 100]);
% 
% 
% subplot(2,3,3); % Change subplot to accommodate theta wave
% plot(t1, beta_filtered7);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kenh thu 7 Beta Wave - Time Domain');
% ylim ([-100 100]);
% 
% subplot(2,3,4); % Add theta wave subplot
% plot(t1, theta_filtered7);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kenh thu 7 Theta Wave - Time Domain');
% ylim ([-100 100]);
% 
% subplot(2,3,5);
% plot(t1, delta_filtered7);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
% title('kenh thu 7 Delta Wave - Time Domain');
% 
% subplot(2,3,6);
% plot(t1, gamma_filtered7);
% xlabel('Time (s)');
% ylabel('Voltage (uV)');
%     title('kenh thu 7 Gamma Wave - Time Domain');
% 
% %mientanso kenh7
% 
% % Frequency domain signal visualization
% n1 = length(filtered_channel7);
% f1 = (0:n1-1)*(fs/n1);
% 
% alpha_fft7 = fft(alpha_filtered7);
% beta_fft7 = fft(beta_filtered7);
% delta_fft7 = fft(delta_filtered7);
% gamma_fft7 = fft(gamma_filtered7);
% theta_fft7 = fft(theta_filtered7); % Add theta FFT
% 
% figure;
% subplot(2,3,1); % Change subplot to accommodate theta wave
% plot(f1, abs(alpha_fft7)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title('kenh thu 7 Alpha Wave - Frequency Domain');
% xlim([0 30]);
% 
% 
% subplot(2,3,2); % Change subplot to accommodate theta wave
% plot(f1, abs(beta_fft7)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title('kenh thu 7 Beta Wave - Frequency Domain');
% xlim([0 30]);
% 
% subplot(2,3,3); % Change subplot to accommodate theta wave
% plot(f1, abs(theta_fft7)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title(' kenh thu 7 Theta Wave - Frequency Domain');
% xlim([0 8]); % Set x-axis limit to theta band
% 
% subplot(2,3,4); % Add theta wave subplot
% plot(f1, abs(delta_fft7)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title(' kenh thu 7 Delta Wave - Frequency Domain');
% xlim([0 5]);
% 
% subplot(2,3,5);
% plot(f1, abs(gamma_fft7)*1000);
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
% title(' kenh thu 7 Gamma Wave - Frequency Domain');
% xlim([0 100]);




