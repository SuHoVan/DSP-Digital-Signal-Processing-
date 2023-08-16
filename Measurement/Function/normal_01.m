function [out] = normal_01(input_data)
% Demonstrates normalization of a signal.

%----------------------------------------------------------------
% configure & initialize
%----------------------------------------------------------------
% Reference signal.
x = input_data;
a = min(x);
b = max(x);
c = b - a;
out = [];
for n = 1:length(x)
    out(:,n) = (x(:,n) - a)/c;
end
end