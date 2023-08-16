function [coor] = R_det_v2(data, fs)

ecg = data;

%% initial
coor = []; % stored the peaks after dectect

qrs = ceil(50*(10^-3)*fs); % qrs complex
rr_min = ceil(0.15*fs); % minimum distance between two peaks

flag = 65;
begin = flag + 2*fs;
threshold = 3;    %3.8
influence = 0.3;

i_transition = nan;
tw = rr_min + qrs; % window size dectect
a = tw + begin; 
th = zeros(1, length(ecg)); % threshold dectect

y_mean = zeros(1, length(ecg));
yn_reg = zeros(1, flag);
signals = zeros(1, length(ecg));
calc_std = zeros(1, length(ecg));

 cal = [0 0 0];

tmp_peak_value = 0;
tmp_peak_indices = 1;

for i = 1:length(ecg)
    %% calculate  mean and standard deviation    
    if (i > begin)
            calc_std(i) = mean([calc_std(i-flag+1:i-1) std(yn_reg)]);
            y_upper(i) = y_mean(i-1) + threshold*calc_std(i-1);
            y_lower(i) = y_mean(i-1) - threshold*calc_std(i-1);
            if abs(ecg(i) - y_mean(i-1)) > threshold*calc_std(i-1)
                if (ecg(i) > y_upper(i))
                    signals(i) = (ecg(i) - y_mean(i-1))/calc_std(i-1);
                    ecg(i) = influence*ecg(i) + (1-influence)*y_mean(i-1);
                elseif (ecg(i) < y_lower(i))
                    signals(i) = abs(ecg(i) - y_mean(i-1))/calc_std(i-1);
                    ecg(i) = influence*ecg(i) - (1-influence)*y_mean(i-1);
                end
            end
            y_mean(i) = mean(yn_reg);
            %% detect peak
            if i <= a
                if (signals(i) >= tmp_peak_value) && (signals(i) ~= inf)
                    tmp_peak_value = signals(i);
                    tmp_peak_indices = i;
                else
                    th(i) = tmp_peak_value;
                end
            elseif (a < i) && (i <= a + rr_min)
                th(i) = tmp_peak_value;
            elseif (i > a + rr_min)
                th(i) = th(i-1)*exp(-6.1/fs);
                if(th(i) < signals(i))
                    if ~isnan(i_transition)
                        cal(3) = abs(tmp_peak_indices - i_transition + rr_min);
                        tw = median(cal);
                    end
                    cal = [cal(2:3) tw];
                    i_transition = i;
                    tmp_peak_value = th(i);
                    a = i + tw;
                    coor = [coor tmp_peak_indices];
                end
            end
    end
        %% Update yn_reg
        yn_reg = [yn_reg(2:flag) ecg(i)];
end

time = [0:length(ecg)-1];
figure('Name', 'R_det : mean and upper curve')
plot(time, data, time, y_mean, 'g-', time, y_upper, 'r-', time, y_lower, 'r-')
figure('Name', 'R_det : peaks and dynamic thresholds ')
plot(time, signals, time, th, 'r')
end