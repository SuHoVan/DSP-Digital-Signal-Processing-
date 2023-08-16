function [dt_data] = remove_dt(t,data)
opol = 6;
[p,s,mu] = polyfit(t,data,opol);
f_y = polyval(p,t,[],mu);
dt_data = data - f_y;
end