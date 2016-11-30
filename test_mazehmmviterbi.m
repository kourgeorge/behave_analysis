function [ output_args ] = test_mazehmmviterbi( input_args )
%TEST_MAZEHMMVITERBI Summary of this function goes here
%   Detailed explanation goes here


test1()

end

function [trR, trNR, eH, eT] = get_real_parameters()
eps = 0.05;
eH = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

eT = [1-eps eps; %o1l2 o2l1
    eps 1-eps;
    eps 1-eps;
    1-eps eps];


trR = [0.7 0.1 0.1 0.1;
    0.1 0.7 0.1 0.1;
    0.1 0.1 0.7 0.1;
    0.7 0.1 0.1 0.7];


trNR = [0.1 0.3 0.3 0.3;
    0.3 0.1 0.3 0.3;
    0.3 0.3 0.1 0.3;
    0.3 0.3 0.3 0.1];

end

function testhmmviterbi()
[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();
[seq,states] = hmmgenerate(1500, realTRr, realEhomo);
estimatedStates = hmmviterbi(seq,realTRr,realEhomo);

bitarray = estimatedStates==states;
Dhmm = sum(bitarray)/1500;

end

function test1()

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

env_type_frac = 0.5;
[envtype,emissions, states, rewards] = ...
    mazehmmgenerate(1500, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);

[currentState, logP] = mazehmmviterbi(emissions,envtype, rewards, realTRr, realTRnr,realEhomo,realEhomo);

bitarray = currentState==states;
D = sum(bitarray)/1500;


end