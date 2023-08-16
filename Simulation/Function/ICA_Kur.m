function [data] = ICA_Kur(data1,Fs)
%% function [data] = ICA_Kur(data1,Fs)
% ICA Kurtosis implementation
%% Inputs
% data1 : nxm matrix input signal: 
%           n : Number of input ICA signal
%           m : Signal length in sample
% Fs : Sample frequency
%% Outputs
% data : nxm matrix ICA-ed signal:
%           n : Number of output ICA signal
%           m : Signal length in sample

[chans frames] = size(data1); % determine the data size
urchans = chans;  % remember original data channels 
% datalength = frames;
%%%%%%%%%%%%%%%%%%%%%% Declare defaults used below %%%%%%%%%%%%%%%%%%%%%%%%
%
wts_blowup = 0;
MAX_WEIGHT           = 1e8;       % guess that weights larger than this have blown up
DEFAULT_STOP         = 0.000001;  % stop training if weight changes below this
DEFAULT_ANNEALDEG    = 60;        % when angle change reaches this value,
DEFAULT_ANNEALSTEP   = 0.90;      %     anneal by multiplying lrate by this
DEFAULT_EXTANNEAL    = 0.98;      %     or this if extended-ICA
DEFAULT_MAXSTEPS     = 512;       % ]top training after this many steps 
DEFAULT_MOMENTUM     = 0.0;       % default momentum weight

DEFAULT_BLOWUP       = 1000000000.0;   % = learning rate has 'blown up'
DEFAULT_BLOWUP_FAC   = 0.8;       % when lrate 'blows up,' anneal by this fac
DEFAULT_RESTART_FAC  = 0.9;       % if weights blowup, restart with lrate
                                  % lower by this factor
MIN_LRATE            = 0.000001;  % if weight blowups make lrate < this, quit
MAX_LRATE            = 0.1;       % guard against uselessly high learning rate
DEFAULT_LRATE        = 0.00065/log(chans); 
                                  % heuristic default - may need adjustment
                                  %   for large or tiny data sets!
% DEFAULT_BLOCK        = floor(sqrt(frames/4));  % heuristic default 
DEFAULT_BLOCK          = ceil(min(5*log(frames),0.3*frames)); % heuristic 
                                  % - may need adjustment!
% Extended-ICA option:
DEFAULT_EXTENDED     = 0;         % default off
DEFAULT_EXTBLOCKS    = 1;         % number of blocks per kurtosis calculation
DEFAULT_NSUB         = 1;         % initial default number of assumed sub-Gaussians
                                  % for extended-ICA
DEFAULT_EXTMOMENTUM  = 0.5;       % momentum term for computing extended-ICA kurtosis
MAX_KURTSIZE         = 6000;      % max points to use in kurtosis calculation
MIN_KURTSIZE         = 2000;      % minimum good kurtosis size (flag warning)
SIGNCOUNT_THRESHOLD  = 25;        % raise extblocks when sign vector unchanged
                                  % after this many steps
SIGNCOUNT_STEP       = 2;         % extblocks increment factor 

DEFAULT_SPHEREFLAG   = 'on';      % use the sphere matrix as the default
                                  %   starting weight matrix
DEFAULT_INTERRUPT    = 'off';     % figure interruption
DEFAULT_PCAFLAG      = 'off';     % don't use PCA reduction
DEFAULT_POSACTFLAG   = 'off';     % don't use posact(), to save space -sm 7/05
DEFAULT_VERBOSE      = 1;         % write ascii info to calling screen
DEFAULT_BIASFLAG     = 1;         % default to using bias in the ICA update rule
DEFAULT_RESETRANDOMSEED = true;   % default to reset the random number generator to a 'random state'

%                                 
%%%%%%%%%%%%%%%%%%%%%%% Set up keyword default values %%%%%%%%%%%%%%%%%%%%%%%%%
%
epochs = 1;							 % do not care how many epochs in data

pcaflag    = DEFAULT_PCAFLAG;
sphering   = DEFAULT_SPHEREFLAG;     % default flags
posactflag = DEFAULT_POSACTFLAG;
verbose    = DEFAULT_VERBOSE;
logfile    = [];

block      = DEFAULT_BLOCK;          % heuristic default - may need adjustment!
lrate      = DEFAULT_LRATE;
annealdeg  = DEFAULT_ANNEALDEG;
annealstep = 0.98;                      % defaults declared below
nochange   = NaN;
momentum   = DEFAULT_MOMENTUM;
maxsteps   = DEFAULT_MAXSTEPS;

weights    = 0;                      % defaults defined below
ncomps     = chans;
biasflag   = DEFAULT_BIASFLAG;

interrupt  = DEFAULT_INTERRUPT;
extended   = DEFAULT_EXTENDED;
extblocks  = DEFAULT_EXTBLOCKS;
kurtsize   = MAX_KURTSIZE;
signsbias  = 0.02;                   % bias towards super-Gaussian components
extmomentum= DEFAULT_EXTMOMENTUM;    % exp. average the kurtosis estimates
nsub       = DEFAULT_NSUB;
wts_blowup = 0;                      % flag =1 when weights too large
wts_passed = 0;                      % flag weights passed as argument
reset_randomseed = DEFAULT_RESETRANDOMSEED;

% 
% adjust nochange if necessary
%
if isnan(nochange) 
    if ncomps > 32
        nochange = 1E-7;
        nochangeupdated = 1; % for fprinting purposes
    else
        nochangeupdated = 1; % for fprinting purposes
        nochange = DEFAULT_STOP;
    end
else 
    nochangeupdated = 0;
end
weights    = 0;                      % defaults defined below
ncomps     = chans;
%%
%%%%%%%%%%%%%%%%% Remove overall row means of data %%%%%%%%%%%%%%%%%%%%%%%
%
% icaprintf(verb,fid,'Removing mean of each channel ...\n');

%BLGBLGBLG replaced
% rowmeans = mean(data');
% data = data - rowmeans'*ones(1,frames);      % subtract row means
%BLGBLGBLG replacement starts
rowmeans = mean(data1,2); %BLG
% data = data - rowmeans'*ones(1,frames);      % subtract row means
data = [];
for iii=1:size(data1,1) %avoids memory errors BLG
    data(iii,:)=data1(iii,:)-rowmeans(iii);
end
%BLGBLGBLG replacement ends
fprintf('Final training data range: %g to %g\n', min(min(data)),max(max(data)));
% figure
% subplot(2,1,1)
% plot(t,data(1,:)); title('remove mean');
% subplot(2,1,2)
% plot(t,data(2,:))
%%
%%%%%%%%%%%%%%%%%%% Perform PCA reduction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     icaprintf(verb,fid,'Reducing the data to %d principal dimensions...\n',ncomps);
%     
%     BLGBLGBLG replaced
%     [eigenvectors,eigenvalues,data] = pcsquash(data,ncomps);
%     make data its projection onto the ncomps-dim principal subspace
%     BLGBLGBLG replacement starts
%     [eigenvectors,eigenvalues,data] = pcsquash(data,ncomps);
%     no need to re-subtract row-means, it was done a few lines above!
    PCdat2 = data';                    % transpose data
    [PCn,PCp]=size(PCdat2);                  % now p chans,n time points
    PCdat2=PCdat2/PCn;
    PCout_test = cov(data');
    PCout=data*PCdat2;                       % Covariance matrix
%     clear PCdat2;
    
    [PCV,PCD] = eig(PCout);                  % get eigenvectors/eigenvalues
    [PCeigenval,PCindex] = sort(diag(PCD));
    PCindex=rot90(rot90(PCindex));
    PCEigenValues=rot90(rot90(PCeigenval))';
    PCEigenVectors=PCV(:,PCindex);
    %PCCompressed = PCEigenVectors(:,1:ncomps)'*data;
    data_pca = PCEigenVectors(:,1:ncomps)'*data;    % Uncorrelated signals
    
    eigenvectors=PCEigenVectors;
    eigenvalues=PCEigenValues;  % #ok<NASGU>
    
%     clear PCn PCp PCout PCV PCD PCeigenval PCindex PCEigenValues PCEigenVectors
%     BLGBLGBLG replacement ends

%%
%%%%%%%%%%%%%%%%%%% Perform sphering %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
sphere = 2.0*inv(sqrtm(double(cov(data')))); % find the "sphering" matrix = spher()
sphere1 = 2.0*inv(sqrtm(double(cov(data_pca'))));
weights = eye(ncomps,chans); % begin with the identity matrix
data = sphere*data; % decorrelate the electrode signals by 'sphereing' them
data_scale = sphere1*data_pca;
% figure
% subplot(2,1,1)
% plot(t,data_scale(1,:)); title('scale');
% subplot(2,1,2)
% plot(t,data_scale(2,:));

%% %%%%%%%%%%%%%%%%%%%%%% Initialize ICA training %%%%%%%%%%%%%%%%%%%%%%%%%
%
nsub = 1;
maxsteps = 512;
block = ceil(min(5*log(frames),0.3*frames));
extblocks = 1;
datalength = frames;
lastt=fix((datalength/block-1)*block+1);
BI=block*eye(ncomps,ncomps);
delta=zeros(1,chans*ncomps);
changes = [];
degconst = 180./pi;
startweights = weights;
prevweights = startweights;
oldweights = startweights;
prevwtchange = zeros(chans,ncomps);
oldwtchange = zeros(chans,ncomps);
lrates = zeros(1,maxsteps);
onesrow = ones(1,block);
bias = zeros(ncomps,1);
signs = ones(1,ncomps);    % initialize signs to nsub -1, rest +1
for k=1:nsub
    signs(k) = -1;
end
signs = diag(signs); % make a diagonal matrix
oldsigns = zeros(size(signs));
signcount = 0;              % counter for same-signs
signcounts = [];
urextblocks = extblocks;    % original value, for resets
old_kk = zeros(1,ncomps);   % for kurtosis momemtumstep=0;

%
%%%%%%%% ICA training loop using the logistic sigmoid %%%%%%%%%%%%%%%%%%%
%
step = 0;
laststep=0;
blockno = 1;  % running block counter for kurtosis interrupts

fprintf('first training step may be slow ...\n');
rand('state',sum(100*clock)); % set the random number generator state to


%%
%%%%%%%%%%%%%%%% Compute Weight %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
biasflag = 1;
extended = 1; 
Value = 10^-3; %varagin
lrate = Value;
momentum = 0; %Default
if biasflag && extended
    while step < maxsteps, %%% ICA step = pass through all the data %%%%%%%%%
        timeperm=randperm(datalength); % shuffle data order at each step

        for t=1:block:lastt, %%%%%%%%% ICA Training Block %%%%%%%%%%%%%%%%%%%
            
            %% promote data block (only) to double to keep u and weights double
            u=weights*double(data(:,timeperm(t:t+block-1))) + bias*onesrow;

            y=tanh(u);                                                       
            weights = weights + lrate*(BI-signs*y*u'-u*u')*weights;
            bias = bias + lrate*sum((-2*y)')';  % for tanh() nonlin.

            if momentum > 0 %%%%%%%%% Add momentum %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                weights = weights + momentum*prevwtchange;
                prevwtchange = weights-prevweights;
                prevweights = weights;
            end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            
            MAX_WEIGHT = 1e8;
            
            kurtsize = frames;
            if max(max(abs(weights))) > MAX_WEIGHT
                wts_blowup = 1;
                change = nochange;
            end
            if ~wts_blowup
                %
                %%%%%%%%%%% Extended-ICA kurtosis estimation %%%%%%%%%%%%%%%%%%%%%
                %while step < maxsteps
                if extblocks > 0 && rem(blockno,extblocks) == 0,
                    % recompute signs vector using kurtosis
                    if kurtsize < frames % 12-22-99 rand() size suggestion by M. Spratling
                        rp = fix(rand(1,kurtsize)*datalength);  % pick random subset
                        % Account for the possibility of a 0 generation by rand
                        ou = find(rp == 0);
                        while ~isempty(ou) % 1-11-00 suggestion by J. Foucher
                            rp(ou) = fix(rand(1,length(ou))*datalength);
                            ou = find(rp == 0);
                        end
                        partact=weights*double(data(:,rp(1:kurtsize)));
                    else                                        % for small data sets,
                        partact=weights*double(data);           % use whole data
                    end
                    m2=mean(partact'.^2).^2;
                    m4= mean(partact'.^4);
                    kk= (m4./m2)-3.0;                           % kurtosis estimates
                    
                    if extmomentum
                        kk = extmomentum*old_kk + (1.0-extmomentum)*kk; % use momentum
                        old_kk = kk;
                    end
                    signs=diag(sign(kk+signsbias));             % pick component signs
                    if signs == oldsigns,
                        signcount = signcount+1;
                    else
                        signcount = 0;
                    end
                    oldsigns = signs;
                    signcounts = [signcounts signcount];
                    if signcount >= SIGNCOUNT_THRESHOLD,
                        extblocks = fix(extblocks * SIGNCOUNT_STEP);% make kurt() estimation
                        signcount = 0;                             % less frequent if sign
                    end                                         % is not changing
                end % extblocks > 0 & . . .
            end % if extended & ~wts_blowup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            blockno = blockno + 1;
            if wts_blowup
                break
            end
        end % for t=1:block:lastt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if ~wts_blowup
            oldwtchange = weights-oldweights;
            step=step+1;
            %
            %%%%%%% Compute and print weight and update angle changes %%%%%%%%%
            %
            lrates(1,step) = lrate;
            angledelta=0.;
            delta=reshape(oldwtchange,1,chans*ncomps);
            change=delta*delta';
        end
        %
        %%%%%%%%%%%%%%%%%%%%%% Restart if weights blow up %%%%%%%%%%%%%%%%%%%%
        %
        if wts_blowup || isnan(change)|isinf(change),  % if weights blow up,
            icaprintf(verb,fid,'');
            step = 0;                          % start again
            change = nochange;
            wts_blowup = 0;                    % re-initialize variables
            blockno = 1;
            lrate = lrate*DEFAULT_RESTART_FAC; % with lower learning rate
            weights = startweights;            % and original weight matrix
            oldweights = startweights;
            change = nochange;
            oldwtchange = zeros(chans,ncomps);
            delta=zeros(1,chans*ncomps);
            olddelta = delta;
            extblocks = urextblocks;
            prevweights = startweights;
            prevwtchange = zeros(chans,ncomps);
            lrates = zeros(1,maxsteps);
            bias = zeros(ncomps,1);

            signs = ones(1,ncomps);    % initialize signs to nsub -1, rest +1
            for k=1:nsub
                signs(k) = -1;
            end
            signs = diag(signs); % make a diagonal matrix
            oldsigns = zeros(size(signs));;

            if lrate> MIN_LRATE
                r = rank(data); % determine if data rank is too low 
                if r<ncomps
%                     icaprintf(verb,fid,'Data has rank %d. Cannot compute %d components.\n',...
%                         r,ncomps);
                    return
                else
%                     icaprintf(verb,fid,...
%                         'Lowering learning rate to %g and starting again.\n',lrate);
                end
            else
%                 icaprintf(verb,fid, ...
%                     'runica(): QUITTING - weight matrix may not be invertible!\n');
                return;
            end
        else % if weights in bounds
            %
            %%%%%%%%%%%%% Print weight update information %%%%%%%%%%%%%%%%%%%%%%
            %
            if step> 2
                angledelta=acos((delta*olddelta')/sqrt(change*oldchange));
            end
            
            verb  = verbose;
            fid = [];
            places = -floor(log10(nochange));
            %
            fprintf('step %d - lrate %5f, wchange %8.8f, angledelta %4.1f deg\n',step,  lrate, change, degconst*angledelta);
            %%%%%%%%%%%%%%%%%%%% Save current values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            changes = [changes change];
            oldweights = weights;
            %
            %%%%%%%%%%%%%%%%%%%% Anneal learning rate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            if degconst*angledelta > annealdeg,
                lrate = lrate*annealstep;          % anneal learning rate
                olddelta   = delta;                % accumulate angledelta until
                oldchange  = change;               %  annealdeg is reached
            elseif step == 1                     % on first step only
                olddelta   = delta;                % initialize
                oldchange  = change;
            end
            
            %
            %%%%%%%%%%%%%%%%%%%% Apply stopping rule %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            if step >2 && change < nochange,      % apply stopping rule
                laststep=step;
                step=maxsteps;                  % stop when weights stabilize
            elseif change > DEFAULT_BLOWUP,      % if weights blow up,
                lrate=lrate*DEFAULT_BLOWUP_FAC;    % keep trying
            end;                                 % with a smaller learning rate
        end; % end if weights in bounds

    end; % end while step < maxsteps (ICA Training) %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%% Finalize Computed Data for Output
  
if strcmpi(interrupt, 'on')
    close(fig);
end

  if ~laststep
    laststep = step;
  end
  lrates = lrates(1,1:laststep);           % truncate lrate history vector

  %
  %%%%%%%%%%%%%% Orient components towards max positive activation %%%%%%
  %
      ser = sphere.*eigenvectors(:,1:ncomps)'.*rowmeans';
          for r = 1:ncomps
              data(r,:) = data(r,:)+ser(r); % add back row means 
          end
          data = weights*data; % OK in single
          
  if ncomps == urchans % if weights are square . . .
      winv = inv(weights*sphere);
  else
      icaprintf(verb,fid,'Using pseudo-inverse of weight matrix to rank order component projections.\n');
      winv = pinv(weights*sphere);
  end
   %
  % compute variances without backprojecting to save time and memory -sm 7/05
  %
  meanvar = sum(winv.^2).*sum((data').^2)/((chans*frames)-1); % from Rey Ramirez 8/07
  %
  %%%%%%%%%%%%%% Sort components by mean variance %%%%%%%%%%%%%%%%%%%%%%%%
  %
    [sortvar, windex] = sort(meanvar);
  windex = windex(ncomps:-1:1); % order large to small 
  meanvar = meanvar(windex);
  %
  %%%%%%%%%%%% re-orient max(abs(activations)) to >=0 ('posact') %%%%%%%%
  %
  weights = weights(windex,:);% reorder the weight matrix
  bias  = bias(windex);       % reorder them
  signs = diag(signs);        % vectorize the signs matrix
  signs = signs(windex);      % reorder them
end

