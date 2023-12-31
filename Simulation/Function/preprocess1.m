function y = preprocess1(x)
%  Preprocess input x
%    This function expects an input vector x.

% Generated by MATLAB(R) 9.11 and Signal Processing Toolbox 8.7.
% Generated on: 11-Jun-2023 20:16:28
%4 4 

%tim 13 8
%5 7
%13 3
y = wdenoise(x, 5, ...
    'Wavelet', 'db3', ...
    'DenoisingMethod', 'Bayes', ...
    'ThresholdRule', 'Median', ...
    'NoiseEstimate', 'LevelIndependent');
end