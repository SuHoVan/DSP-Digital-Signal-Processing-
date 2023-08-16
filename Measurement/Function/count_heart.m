%% Detect R-peak
clear all; close all; clc;
fs = 250;
sig = load("D:\EEG\ecg_bandpass.csv", '-ascii');
t = 1:length(sig);

[coor] = R_det_v2(sig, fs);

% opt = T;
% detect_result = R_det(sig, opt, false);
% 
% coor = detect_result.coor;
figure('Name', 'Detecting result')
plot(t, sig, 'b',(coor), sig(coor), 'o');


%% evaluate algolithm
coor = unique(coor);
coor = coor';
ann_correct = [];
for coor_ = 1:length(coor)
    ann_sub = abs(ann_correct - coor(coor_));
    ann_con = ann_sub < 10;
    ann_correct = [ann_correct ann(ann_con)];
end
ann_incorrect = setxor(ann, ann_correct);
figure('Name', 'compare')
plot(t, sig, 'b',ann_incorrect, sig(ann_incorrect), 'ro', ann_correct, sig(ann_correct), 'go');

coor_correct = [];
for ann_ = 1:length(ann)
    coor_sub = abs(coor - ann(ann_));
    coor_con = coor_sub < 10;
    coor_correct = [coor_correct coor(coor_con)];
end
coor_incorrect = setxor(coor, coor_correct);
