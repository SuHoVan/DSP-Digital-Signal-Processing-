function [qrs_ls, qrs] = calc_qrs(signal, locs)
    left = 0;
    right = 0;
    qrs = [];
    qrs_ls = [];
    for loc = locs
        ii_r = loc;
        ii_l = loc;
        bk_1 = false;
        bk_2 = false;
        while ~bk_1 && ~bk_2
            if (signal(ii_l) > signal(ii_l - 1)) || ii_l == loc
                ii_l = ii_l - 2;
            else
                bk_1 = true;
            end
    
            if (signal(ii_r) > signal(ii_r + 1)) || ii_r == loc
                ii_r = ii_r + 2;
            else
                bk_2 = true;
            end  
        end
        left = ii_l;
        right = ii_r;
        qrs_ls = [qrs_ls left right];

        qrs = [qrs right-left];
        left = 0;
        right = 0;
    end
end