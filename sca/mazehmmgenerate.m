function [envtype, seq, states,rewards ] = mazehmmgenerate(L, TRReward, TRNoReward, eHomo, eHetro, reward_rule_transition)
%SYNTHETIC_DATA Summary of this function goes here
%   Detailed explanation goes here

rule1 = [0 1; 1 0];
rule2 = [1 0; 1 0];
rule3 = [0 1; 0 1];
rule4 = [1 0; 1 0];

rules = {rule1, rule2, rule3, rule4};

numStates = size(TRReward,1);
numEmissions = size(eHomo,2);

% create two random sequences, one for state changes, one for emission
statechange = rand(1,L);
randvals = rand(1,L);
envtypechange = rand(1,L);
rule_change = rand(1,L);

% calculate cumulative probabilities
trcReward = cumsum(TRReward,2);
trcNoReward = cumsum(TRNoReward,2);
ecHomo = cumsum(eHomo,2);
ecHetro = cumsum(eHetro,2);

% normalize these just in case they don't sum to 1.
trcReward = trcReward./repmat(trcReward(:,end),1,numStates);
trcNoReward = trcNoReward./repmat(trcNoReward(:,end),1,numStates);
ecHomo = ecHomo./repmat(ecHomo(:,end),1,numEmissions);
ecHetro = ecHetro./repmat(ecHetro(:,end),1,numEmissions);


currentstate = 1;
seq = zeros(1,L);
states = zeros(1,L);
envtype = zeros(1,L);
rewards = zeros(1,L);
curr_rewarded_rule = 1;
for count = 1:L
    % determine environment type
    if (envtypechange(count) < 0.5)
        currentconf = 1;
    else
        currentconf = 2;
    end
    
    if (count == 1)
        lastreward = 0; %no reward at trial 0
    else
        lastreward = (rewards(count-1) == 1);
    end
    
    % calculate state transition
    if (lastreward == true)
        trc = trcReward;
    else
        trc = trcNoReward;
    end
    stateVal = statechange(count);
    state = 1;
    for innerState = numStates-1:-1:1
        if stateVal > trc(currentstate,innerState)
            state = innerState + 1;
            break;
        end
    end
    
    % calculate emission
    if (currentconf == 1)
        ec = ecHomo;
    else
        ec = ecHetro;
    end
    val = randvals(count);
    emit = 1;
    for inner = numEmissions-1:-1:1
        if val  > ec(state,inner)
            emit = inner + 1;
            break
        end
    end
    
    % the rule change
%     if (rule_change(count)< reward_rule_transition)
%         curr_rewarded_rule = randi([1 4]);
%     end
    
    if (mod(count,10)==0)
        curr_rewarded_rule = randi([1 4]);
    end
    curr_rewarded_rule = 1;
    
    % add values and states to output
    seq(count) = emit;
    states(count) = state;
    envtype(count) = currentconf;
    rewards(count) = rules{curr_rewarded_rule}(currentconf,emit);
    currentstate = state;
end