function [dataICA] = ICA(data_mix)
data_mix = single(data_mix);            %pop_checkset (line 747)

% verify the type of the variables
% --------------------------------
% data dimensions -------------------------
nbchan = 0;
if ~isequal(size(data_mix,1), nbchan)
   disp( [ 'eeg_checkset warning: number of columns in data (' int2str(size(data_mix,1)) ...
          ') does not match the number of channels (' int2str(nbchan) '): corrected' ]);
   nbchan = size(data_mix,1);
end

% size of data -----------
trials = 0;
if size(data_mix,3) ~= trials
    disp( ['eeg_checkset warning: 3rd dimension size of data (' int2str(size(data_mix,3)) ...
        ') does not match the number of epochs (' int2str(trials) '), corrected' ]);
    trials = size(data_mix,3);
end

pnts = 0;
if size(data_mix,2) ~= pnts
    disp( [ 'eeg_checkset warning: number of columns in data (' int2str(size(data_mix,2)) ...
        ') does not match the number of points (' int2str(pnts) '): corrected' ]);
    pnts = size(data_mix,2);
end

[row,col] = size(data_mix);
chanind = 1:row;                    %g.chanind
icachansind = chanind;              %EEG.icachansind
n = length(chanind);
pnts = col;                         % length(data_2nd)
trials = 1;
tmpdata = reshape(data_mix(chanind,:,:), length(chanind), pnts*trials);
tmpdata = tmpdata - repmat(mean(tmpdata,2), [1 size(tmpdata,2)]);           % zero mean 
disp('Attempting to convert data matrix to double precision for more accurate ICA results.')
tmpdata = double(tmpdata);
tmpdata = tmpdata - repmat(mean(tmpdata,2), [1 size(tmpdata,2)]); % zero mean (more precise than single precision)
%% ========== Acsobiro ============
[icawinv,Dx] = acsobiro(tmpdata,n);
%H = icawinv
%S = act
icaweights = pinv(icawinv);
icasphere = eye(size(icaweights,2));

% Reorder components by variance
meanvar = sum(icawinv.^2).*sum(transpose((icaweights *  icasphere)*data_mix(icachansind,:)).^2)/((length(icachansind)*pnts)-1);
[~, windex] = sort(meanvar);
windex = windex(end:-1:1);                                % order large to small
meanvar = meanvar(windex);
icaweights = icaweights(windex,:);
icawinv    = pinv(icaweights * icasphere);

icawinv    = double(icawinv);                             % required for dipole fitting, otherwise it crashes
icaweights = double(icaweights);
icasphere  = double(icasphere);

if mean(mean(abs(pinv(icaweights * icasphere) - icawinv))) < 0.0001
    disp('Scaling components to RMS microvolt');
    scaling = repmat(sqrt(mean(icawinv(:,:).^2))', [1 size(icaweights,2)]);

    icaweights = icaweights .* scaling;
    icawinv = pinv(icaweights * icasphere);
end

% ALLEEG = eeg_store(ALLEEG, EEG, g.dataset); pop_runica(line 598)
% eeg_store(line 111)
% if isempty(varargin) ... no text output and no check (study loading)
% ->eeg_checkset

if mean(mean(abs(pinv(icaweights * icasphere) - icawinv))) < 0.0001
    disp('Scaling components to RMS microvolt');
    scaling = repmat(sqrt(mean(icawinv(:,:).^2))', [1 size(icaweights,2)]);

    icaweights = icaweights .* scaling;
    icawinv = pinv(icaweights * icasphere);
end

%return pop_runica (line 609)
if mean(mean(abs(pinv(icaweights * icasphere) - icawinv))) < 0.0001
    disp('Scaling components to RMS microvolt');
    scaling = repmat(sqrt(mean(icawinv(:,:).^2))', [1 size(icaweights,2)]);

    icaweights = icaweights .* scaling;
    icawinv = pinv(icaweights * icasphere);
end
dataICA = (pinv(icawinv)*data_mix);   %act 
dataICA = double(dataICA);          %double is more accurate, default for calculating
end