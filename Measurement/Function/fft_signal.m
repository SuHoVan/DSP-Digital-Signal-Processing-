function [f, P1] = fft_signal(data, Fs)
fft_data = fft(data);
L = length(data);
P2 = abs(fft_data/L);   %Compute the two-sided spectrum P2
P1 = P2(1:L/2+1);       %Compute the single-sided spectrum P1 based on P2 
P1(2:end-1) = 2*P1(2:end-1);    %The even-valued signal length L.
f = Fs*(0:(L/2))/L;
end