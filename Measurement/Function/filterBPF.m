function data_out = filterBPF(data_in,Fs,fc1,fc2,order,type,mode)
switch type
    case 1
        [b,a] = butter(order,[fc1/(Fs/2) fc2/(Fs/2)],mode);
    case 2
        [b,a] = ellip(order, [fc1/(Fs/2) fc2/(Fs/2)], mode);
    case 3
        [b,a] = cheby1(order,0.1,[fc1/(Fs/2) fc2/(Fs/2)], mode);
end
%         data_out = filter(b,a,data_in); 
%         grpdelay([b,a],order,'whole'); 
%         delay = mean(grpdelay([b;a],order,'whole'));
%         data_out = data_out(1:end - delay);
        h = fvtool(b,a);
        data_out = filtfilt(b,a,data_in);
end