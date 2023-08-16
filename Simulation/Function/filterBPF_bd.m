function data_out = filterBPF_bd(data_in,Fs,fc1,fc2,order,type,mode)
switch type
    case 1
        [b,a] = butter(order,[fc1/(Fs/2) fc2/(Fs/2)],mode);
    case 2
        [b,a] = ellip(order,[fc1/(Fs/2) fc2/(Fs/2)],mode);
end

        data_out = filter(b,a,data_in);
        
end