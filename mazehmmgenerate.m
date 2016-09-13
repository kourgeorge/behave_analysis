function [envtype,seq,states,rewards ] = mazehmmgenerate(L, TRReward, TRNoReward, eHomo, eHetro, envtypefrac, RewardedStates)
%SYNTHETIC_DATA Summary of this function goes here
%   Detailed explanation goes here

numStates = size(TRReward,1);
numEmissions = size(eHomo,2);

% create two random sequences, one for state changes, one for emission
statechange = rand(1,L);
randvals = rand(1,L);
envtypechange = rand(1,L);

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


currentstate = 5;
seq = zeros(1,L);
states = zeros(1,L);
envtype = zeros(1,L);
rewards = zeros(1,L);

for count = 1:L
    % determine environment type
    if (envtypechange(count) < envtypefrac)
        currentconf = 1;
    else
        currentconf = 2;
    end
    
    if (count == 1)
        lastreward = 0; %no reward at trial 0
    else
        lastreward = (seq(count-1) == 1);
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
    % add values and states to output
    seq(count) = emit;
    states(count) = state;
    envtype(count) = currentconf;
    rewards(count) = any(seq(count) == RewardedStates);
    currentstate = state;
end
        

