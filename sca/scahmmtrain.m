function [guessTRr, guessTRnr, guessPolicies, logliks] = scahmmtrain(actions, envstates, rewards, guessTRr, guessTRnr, guessPolicies, varargin)

tol = 1e-5;
trtol = tol;
etol = tol;
maxiter = 200;
pseudoEcounts = false;
pseudoTRcounts = false;
verbose = false;
[numPolicies, checkTr] = size(guessTRr);
if checkTr ~= numPolicies
    error(message('stats:hmmtrain:BadTransitions'));
end

% number of rows of e must be same as number of states

numActions = guessPolicies{1}.outputs{2}.size;
numStates = guessPolicies{1}.inputs{1}.size;
initialPolicies = guessPolicies;
% if checkE ~= numStates
%     error(message('stats:hmmtrain:InputSizeMismatch'));
% end
% if (numStates ==0 || numEmissions == 0)
%     guessTRr = [];
%     guessEhomo = [];
%     return
% end

baumwelch = true;

% if nargin > 5
%     %if rem(nargin,2)== 0
%     %    error(message('stats:hmmtrain:WrongNumberArgs', mfilename));
%     %end
%     okargs = {'symbols','tolerance','pseudoemissions','pseudotransitions','maxiterations','verbose','algorithm','trtol','etol'};
%     dflts  = {[]        []         []                []                  maxiter         verbose   ''           []      []};
%     [symbols,tol,pseudoEhomo,pseudoTRr,maxiter,verbose,alg,trtol,etol] = ...
%         internal.stats.parseArgs(okargs, dflts, varargin{:});
%     
%     
%     if ~isempty(symbols)
% %         numSymbolNames = numel(symbols);
% %         if ~isvector(symbols) || numSymbolNames ~= numEmissions
% %             error(message('stats:hmmtrain:BadSymbols', numEmissions));
% %         end
% %         
%         % deal with a single sequence first
%         if ~iscell(seqs) || ischar(seqs{1})
%             [~, seqs]  = ismember(seqs,symbols);
%             if any(seqs(:)==0)
%                 error(message('stats:hmmtrain:MissingSymbol'));
%             end
%         else  % now deal with a cell array of sequences
%             numSeqs = numel(seqs);
%             newSeqs = cell(numSeqs,1);
%             for count = 1:numSeqs
%                 [~, newSeqs{count}] = ismember(seqs{count},symbols);
%                 if any(newSeqs{count}(:)==0)
%                     error(message('stats:hmmtrain:MissingSymbol'));
%                 end
%             end
%             seqs = newSeqs;
%         end
%     end
%     if ~isempty(pseudoEhomo)
%         [rows, cols] = size(pseudoEhomo);
%         if  rows < numStates
%             error(message('stats:hmmtrain:BadPseudoEmissionsRows'));
%         end
%         if  cols < numEmissions
%             error(message('stats:hmmtrain:BadPseudoEmissionsCols'));
%         end
%         numStates = rows;
%         numEmissions = cols;
%         pseudoEcounts = true;
%         pseudoEhetro = pseudoEhomo;
%     end
%     if ~isempty(pseudoTRr)
%         [rows, cols] = size(pseudoTRr);
%         if rows ~= cols
%             error(message('stats:hmmtrain:BadPseudoTransitions'));
%         end
%         if  rows < numStates
%             error(message('stats:hmmtrain:BadPseudoEmissionsSize'));
%         end
%         numStates = rows;
%         pseudoTRcounts = true;
%     end
%     if ischar(verbose)
%         verbose = any(strcmpi(verbose,{'on','true','yes'}));
%     end
%     
%     if ~isempty(alg)
%         alg = internal.stats.getParamVal(alg,{'baumwelch','viterbi'},'Algorithm');
%         baumwelch = strcmpi(alg,'baumwelch');
%     end
% end

if isempty(tol)
    tol = 1e-6;
end
if isempty(trtol)
    trtol = tol;
end
if isempty(etol)
    etol = tol;
end


if isnumeric(actions)
    [numSeqs, seqLength] = size(actions);
    cellflag = false;
elseif iscell(actions)
    numSeqs = numel(actions);
    cellflag = true;
else
    error(message('stats:hmmtrain:BadSequence'));
end

% initialize the counters
TRr = zeros(numPolicies);
TRnr = zeros(numPolicies);

if ~pseudoTRcounts
    pseudoTRr = TRr;
    pseudoTRnr = TRnr;
end
policies = guessPolicies;

if ~pseudoEcounts
    pseudopolicies = policies;
end

converged = false;
loglik = 1; % loglik is the log likelihood of all sequences given the TR and E
logliks = zeros(1,maxiter);
for iteration = 1:maxiter
    
    state_action_probs = tabulate_neural_policies(policies,numStates);

    oldLL = loglik;
    loglik = 0;
    oldGuessTRr = guessTRr;
    oldGuessTRnr = guessTRnr;
    oldGuessPolicies = guessPolicies;
    for count = 1:numSeqs
        if cellflag
            actionseq = actions{count};
            envstateseq = envstates{count};
            rewardseq = rewards{count};
            seqLength = length(actionseq);
        else
            actionseq = actions(count,:);
            envstateseq = envstates(count,:);
            rewardseq = rewards(count,:);
        end
        
        if baumwelch   % Baum-Welch training
            % get the scaled forward and backward probabilities
            [pstates,logPseq,fs,bs,scale] = scahmmdecode(actionseq,envstateseq,rewardseq,guessTRr, guessTRnr,guessPolicies);
            
            loglik = loglik + logPseq;
            logf = log(abs(fs));
            logb = log(abs(bs));
            logGTRr = log(guessTRr);
            logGTRnr = log(guessTRnr);
            % f and b start at 0 so offset seq by one
            actionseq = [0 actionseq];
            envstateseq = [0 envstateseq];
            rewardseq = [0 rewardseq];
            
            for k = 1:numPolicies
                for l = 1:numPolicies
                    for i = 1:seqLength
                        curr_state_action_prob = log(state_action_probs{envstateseq(i+1)}(l, actionseq(i+1)));
                        %curr_state_actions_prob = log(policies{l}(envstateseq(i+1)));
                        %curr_state_action_prob = curr_state_actions_prob(actionseq(i+1));
                        if (rewardseq(i) == 1)
                            TRr(k,l) = TRr(k,l) + exp( logf(k,i) + logGTRr(k,l) + curr_state_action_prob + logb(l,i+1))./scale(i+1); 
                        else
                            TRnr(k,l) = TRnr(k,l) + exp( logf(k,i) + logGTRnr(k,l) + curr_state_action_prob + logb(l,i+1))./scale(i+1);
                        end
                    end
                end
            end
            
            % optimize the policies by relatively by the probability that the trial i was in policy k 
            for k = 1:numPolicies
%                 for i = 1:numActions
%                     for j= 1:numStates
%                         rel_trials_actions = find(actionseq == i);
%                         state_j_trial = find(envstateseq==j);
%                         pos = intersect(rel_trials_actions,state_j_trial);
%                         policies(k,i) = policies(k,i) + sum(exp(logf(k,pos)+logb(k,pos)));
%                         
%                     end
%                 end
                all_states = onehot(envstateseq',1:numStates)';
                all_actions = onehot(actionseq',1:numActions)';
                policies{k} = train(initialPolicies{k}, all_states, all_actions, [], [], exp(logf(k,:)+logb(k,:))); %(sum(exp(logf(k,pos_homo)+logb(k,pos_homo)))
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
%     totalEmissionsHomo = sum(policies,2);
%     totalEmissionsHetro = sum(Ehetro,2);
    totalTransitionsR = sum(TRr,2);
    totalTransitionsNR = sum(TRnr,2);
    
    % avoid divide by zero warnings
    guessTRr  = TRr./(repmat(totalTransitionsR,1,numPolicies));
    guessTRnr  = TRnr./(repmat(totalTransitionsNR,1,numPolicies));
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
    
    guessPolicies = policies;
  
    % clean up any remaining Nans
    guessTRr(isnan(guessTRr)) = 0;
    guessTRnr(isnan(guessTRnr)) = 0;
    
    if verbose
        if iteration == 1
            fprintf('%s\n',getString(message('stats:hmmtrain:RelativeChanges')));
            fprintf('   Iteration       Log Lik    TransitionReward    TransitionNoReward     Emmission1    Emmission2\n');
        else
            fprintf('  %6d      %12g  %18g  %18g  %12g\n', iteration, ...
                (abs(loglik-oldLL)./(1+abs(oldLL))), ...
                norm(guessTRr - oldGuessTRr,inf)./numPolicies, ...
                norm(guessTRnr - oldGuessTRnr,inf)./numPolicies, ...
                norm(guessPolicies - oldGuessPolicies,inf)./numActions);
        end
    end
    % Durbin et al recommend loglik as the convergence criteria  -- we also
    % use change in TR and E. Use (undocumented) option trtol and
    % etol to set the convergence tolerance for these independently.
    %
    logliks(iteration) = loglik;
    %plot(logliks(find(logliks)))
    if (abs(loglik-oldLL)/(1+abs(oldLL))) < tol
        if norm(guessTRr - oldGuessTRr,inf)/numPolicies < trtol && norm(guessTRnr - oldGuessTRnr,inf)/numPolicies < trtol
           % if (norm(guessEhetro - oldGuessEhetro,inf)/numActions < etol)
                if verbose
                    fprintf('%s\n',getString(message('stats:hmmtrain:ConvergedAfterIterations',iteration)))
                end
                converged = true;
                break
            %end
        end
    end
    
    TRr = pseudoTRr;
    TRnr = pseudoTRnr;
    policies =  pseudopolicies;
    
end
if ~converged
    warning(message('stats:hmmtrain:NoConvergence', num2str( tol ), maxiter));
end
logliks(logliks ==0) = [];
