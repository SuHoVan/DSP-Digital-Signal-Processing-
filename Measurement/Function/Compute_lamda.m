function [lamda, delta] = Compute_lamda(cd,Dnh)
    N = length(cd);
    sigma = median(abs(cd))/Dnh;
    lamda = sigma .* sqrt(2 .* log(N));
    for i = 1:N
        if abs(cd(i)) <= lamda
            delta(i) = 0;
        else 
            delta(i) = cd(i) - (lamda.^2 ./ cd(i));
%             delta(i) = cd(i) - lamda;
%         else 
%             delta(i) = cd(i) + lamda;
        end
    end
end