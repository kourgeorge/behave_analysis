function [guessTRr, guessTRnr, guessEhomo, guessEhetro, logliks] = mazehmmtrain(seqs, exptypes, rewards, guessTRr, guessTRnr, guessEhomo, guessEhetro, varargin)

tol = 1e-6;
trtol = tol;
etol = tol;
maxiter = 500;
pseudoEcounts = false;
pseudoTRcounts = false;
verbose = false;
[numStates, checkTr] = size(guessTRr);
if checkTr ~= numStates
    error(message('stats:hmmtrain:BadTransitions'));
end

% number of rows of e must be same as number of states

[checkE, numEmissions] = size(guessEhomo);
if checkE ~= numStates
    error(message('stats:hmmtrain:InputSizeMismatch'));
end
if (numStates ==0 || numEmissions == 0)
    guessTRr = [];
    guessEhomo = [];
    return
end

baumwelch = true;

if nargin > 5
    %if rem(nargin,2)== 0
    %    error(message('stats:hmmtrain:WrongNumberArgs', mfilename));
    %end
    okargs = {'symbols','tolerance','pseudoemissions','pseudotransitions','maxiterations','verbose','algorithm','trtol','etol'};
    dflts  = {[]        []         []                []                  maxiter         verbose   ''           []      []};
    [symbols,tol,pseudoEhomo,pseudoTRr,maxiter,verbose,alg,trtol,etol] = ...
        internal.stats.parseArgs(okargs, dflts, varargin{:});
    
    
    if ~isempty(symbols)
        numSymbolNames = numel(symbols);
        if ~isvector(symbols) || numSymbolNames ~= numEmissions
            error(message('stats:hmmtrain:BadSymbols', numEmissions));
        end
        
        % deal with a single sequence first
        if ~iscell(seqs) || ischar(seqs{1})
            [~, seqs]  = ismember(seqs,symbols);
            if any(seqs(:)==0)
                error(message('stats:hmmtrain:MissingSymbol'));
            end
        else  % now deal with a cell array of sequences
            numSeqs = numel(seqs);
            newSeqs = cell(numSeqs,1);
            for count = 1:numSeqs
                [~, newSeqs{count}] = ismember(seqs{count},symbols);
                if any(newSeqs{count}(:)==0)
                    error(message('stats:hmmtrain:MissingSymbol'));
                end
            end
            seqs = newSeqs;
        end
    end
    if ~isempty(pseudoEhomo)
        [rows, cols] = size(pseudoEhomo);
        if  rows < numStates
            error(message('stats:hmmtrain:BadPseudoEmissionsRows'));
        end
        if  cols < numEmissions
            error(message('stats:hmmtrain:BadPseudoEmissionsCols'));
        end
        numStates = rows;
        numEmissions = cols;
        pseudoEcounts = true;
        pseudoEhetro = pseudoEhomo;
    end
    if ~isempty(pseudoTRr)
        [rows, cols] = size(pseudoTRr);
        if rows ~= cols
            error(message('stats:hmmtrain:BadPseudoTransitions'));
        end
        if  rows < numStates
            error(message('stats:hmmtrain:BadPseudoEmissionsSize'));
        end
        numStates = rows;
        pseudoTRcounts = true;
    end
    if ischar(verbose)
        verbose = any(strcmpi(verbose,{'on','true','yes'}));
    end
    
    if ~isempty(alg)
        alg = internal.stats.getParamVal(alg,{'baumwelch','viterbi'},'Algorithm');
        baumwelch = strcmpi(alg,'baumwelch');
    end
end

if isempty(tol)
    tol = 1e-6;
end
if isempty(trtol)
    trtol = tol;
end
if isempty(etol)
    etol = tol;
end


if isnumeric(seqs)
    [numSeqs, seqLength] = size(seqs);
    cellflag = false;
elseif iscell(seqs)
    numSeqs = numel(seqs);
    cellflag = true;
else
    error(message('stats:hmmtrain:BadSequence'));
end

% initialize the counters
TRr = zeros(numStates);
TRnr = zeros(numStates);

if ~pseudoTRcounts
    pseudoTRr = TRr;
    pseudoTRnr = TRnr;
end
Ehomo = zeros(numStates,numEmissions);
Ehetro = zeros(numStates,numEmissions);

if ~pseudoEcounts
    pseudoEhomo = Ehomo;
    pseudoEhetro = Ehetro;
end

converged = false;
loglik = 1; % loglik is the log likelihood of all sequences given the TR and E
logliks = zeros(1,maxiter);
for iteration = 1:maxiter
    oldLL = loglik;
    loglik = 0;
    oldGuessEhomo = guessEhomo;
    oldGuessEhetro = guessEhetro;
    oldGuessTRr = guessTRr;
    oldGuessTRnr = guessTRnr;
    for count = 1:numSeqs
        if cellflag
            seq = seqs{count};
            exptype = exptypes{count};
            reward = rewards{count};
            seqLength = length(seq);
        else
            seq = seqs(count,:);
            exptype = exptypes(count,:);
            reward = rewards(count,:);
        end
        
        if baumwelch   % Baum-Welch training
            % get the scaled forward and backward probabilities
            [pstates,logPseq,fs,bs,scale] = mazehmmdecode(seq,exptype,reward,guessTRr, guessTRnr,guessEhomo,guessEhetro);
            
            loglik = loglik + logPseq;
            logf = log(fs);
            logb = log(bs);
            logGEhomo = log(guessEhomo);
            logGEhetro = log(guessEhetro);
            logGTRr = log(guessTRr);
            logGTRnr = log(guessTRnr);
            % f and b start at 0 so offset seq by one
            seq = [0 seq];
            exptype = [0 exptype];
            reward = [0 reward];
            
            for k = 1:numStates
                for l = 1:numStates
                    for i = 1:seqLength
                        if (exptype(i+1)==1)
                            tmplogGE = logGEhomo;
                        else
                            tmplogGE = logGEhetro;
                        end
                        
                        if (reward(i) == 1)
                            TRr(k,l) = TRr(k,l) + exp( logf(k,i) + logGTRr(k,l) + tmplogGE(l,seq(i+1)) + logb(l,i+1))./scale(i+1);
                        else
                            TRnr(k,l) = TRnr(k,l) + exp( logf(k,i) + logGTRnr(k,l) + tmplogGE(l,seq(i+1)) + logb(l,i+1))./scale(i+1);
                        end
                    end
                end
            end
            for k = 1:numStates
                for i = 1:numEmissions
                    rel_trials_emission = find(seq == i);
                    homo_trial = find(exptype==1);
                    hetr_trial = find(exptype==2);
                    pos_homo = intersect(rel_trials_emission,homo_trial);
                    pos_hetro = intersect(rel_trials_emission,hetr_trial);
                    Ehomo(k,i) = Ehomo(k,i) + sum(exp(logf(k,pos_homo)+logb(k,pos_homo)));
                    Ehetro(k,i) = Ehetro(k,i) + sum(exp(logf(k,pos_hetro)+logb(k,pos_hetro)));
                end
            end
            %         else  % Viterbi training
            %             [estimatedStates,logPseq]  = hmmviterbi(seq,guessTR,guessE);
            %             loglik = loglik + logPseq;
            %             % w = warning('off');
            %             [iterTR, iterE] = hmmestimate(seq,estimatedStates,'pseudoe',pseudoE,'pseudoTR',pseudoTR);
            %             %warning(w);
            %             % deal with any possible NaN values
            %             iterTR(isnan(iterTR)) = 0;
            %             iterE(isnan(iterE)) = 0;
            %
            %             TR = TR + iterTR;
            %             E1 = E1 + iterE;
        end
    end
    totalEmissionsHomo = sum(Ehomo,2);
    totalEmissionsHetro = sum(Ehetro,2);
    totalTransitionsR = sum(TRr,2);
    totalTransitionsNR = sum(TRnr,2);
    
    % avoid divide by zero warnings
    guessEhomo = Ehomo./(repmat(totalEmissionsHomo,1,numEmissions));
    guessEhetro = Ehetro./(repmat(totalEmissionsHetro,1,numEmissions));
    guessTRr  = TRr./(repmat(totalTransitionsR,1,numStates));
    guessTRnr  = TRnr./(repmat(totalTransitionsNR,1,numStates));
    % if any rows have zero transitions then assume that there are no
    % transitions out of the state.
    if any(totalTransitionsR == 0)
        noTransitionRows = find(totalTransitionsR == 0);
        guessTRr(noTransitionRows,:) = 0;
        guessTRr(sub2ind(size(guessTRr),noTransitionRows,noTransitionRows)) = 1;
    end
    if any(totalTransitionsNR == 0)
        noTransitionRows = find(totalTransitionsNR == 0);
        guessTRnr(noTransitionRows,:) = 0;
        guessTRnr(sub2ind(size(guessTRnr),noTransitionRows,noTransitionRows)) = 1;
    end
    % clean up any remaining Nans
    guessTRr(isnan(guessTRr)) = 0;
    guessTRnr(isnan(guessTRnr)) = 0;
    guessEhomo(isnan(guessEhomo)) = 0;
    guessEhetro(isnan(guessEhetro)) = 0;
    
    if verbose
        if iteration == 1
            fprintf('%s\n',getString(message('stats:hmmtrain:RelativeChanges')));
            fprintf('   Iteration       Log Lik    TransitionReward    TransitionNoReward     Emmission1    Emmission2\n');
        else
            fprintf('  %6d      %12g  %18g  %18g  %12g  %12g\n', iteration, ...
                (abs(loglik-oldLL)./(1+abs(oldLL))), ...
                norm(guessTRr - oldGuessTRr,inf)./numStates, ...
                norm(guessTRnr - oldGuessTRnr,inf)./numStates, ...
                norm(guessEhomo - oldGuessEhomo,inf)./numEmissions, ...
                norm(guessEhetro - oldGuessEhetro,inf)./numEmissions);
        end
    end
    % Durbin et al recommend loglik as the convergence criteria  -- we also
    % use change in TR and E. Use (undocumented) option trtol and
    % etol to set the convergence tolerance for these independently.
    %
    logliks(iteration) = loglik;
    if (abs(loglik-oldLL)/(1+abs(oldLL))) < tol
        if norm(guessTRr - oldGuessTRr,inf)/numStates < trtol && norm(guessTRnr - oldGuessTRnr,inf)/numStates < trtol
            if (norm(guessEhomo - oldGuessEhomo,inf)/numEmissions < etol) && (norm(guessEhetro - oldGuessEhetro,inf)/numEmissions < etol)
                if verbose
                    fprintf('%s\n',getString(message('stats:hmmtrain:ConvergedAfterIterations',iteration)))
                end
                converged = true;
                break
            end
        end
    end
    Ehomo =  pseudoEhomo;
    Ehetro =  pseudoEhetro;
    TRr = pseudoTRr;
    TRnr = pseudoTRnr;
end
if ~converged
    warning(message('stats:hmmtrain:NoConvergence', num2str( tol ), maxiter));
end
logliks(logliks ==0) = [];
