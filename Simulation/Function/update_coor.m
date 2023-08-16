function [coor_new]  = update_coor(coor_in, rr ,ecg)

    temp_max = ecg(coor_in, 1);
    coor_new = coor_in;

    if coor_in - rr < 0
        indexs = 1:coor_in;
    else
        indexs = coor_in:-1:coor_in-rr;
    end

    for coor_ = indexs
        if ecg(coor_) > temp_max
            coor_new = coor_;
            temp_max = ecg(coor_);
        end
    end

    

end
