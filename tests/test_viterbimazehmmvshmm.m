function test_viterbimazehmmvshmm()
res_hmmprobs = [];
res_mazehmmprobs = [];
res_mazehmmprobsreal = [];
for i=1:5
    [hmmprobs, mazehmmprobs, mazehmmprobsreal]=calcseqposteriorprobability(100,1000,5);
    res_hmmprobs = [res_hmmprobs; hmmprobs];
    res_mazehmmprobs = [res_mazehmmprobs; mazehmmprobs];
    res_mazehmmprobsreal=[res_mazehmmprobsreal;mazehmmprobsreal];
end

norm_mazehmmprobs = res_mazehmmprobs./res_mazehmmprobsreal.*100;
norm_hmmprobs = res_hmmprobs./res_mazehmmprobsreal.*100;


bar(floor(linspace(100,1000,5))', [mean(norm_mazehmmprobs); mean(norm_hmmprobs)]','LineWidth',1.5)
legend('MBW', 'BW')
xlabel('Sequence length')
ylabel('Normalized percentage of correct states estimation')
title('The normalized percentage of correct states estimation using on a sequence of 200 trials')

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

function [hmmprobs, mazehmmprobs, mazehmmprobsreal] = calcseqposteriorprobability(from, to, interval)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();


env_type_frac = 0.5;
[envtype,emissions, states, rewards] = ...
    mazehmmgenerate(1500, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);

[testseq.envtype,testseq.emissions, testseq.states, testseq.rewards] = ...
    mazehmmgenerate(200, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);


guess.trr = getrandomdistribution(4,4);
guess.trnr = getrandomdistribution(4,4);
guess.eh = getrandomdistribution(4,2);
guess.et = getrandomdistribution(4,2);


max_iter = 1500;
mazehmmprobs = [];
hmmprobs=[];
mazehmmprobsreal=[];
lengths = linspace(from,to, interval);

for seq_length=floor(lengths)
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    [mazehmmestimate,hmmestimate] = train_models(seq_data, guess, max_iter);
    
    [correctstates_mazehmm,correctstates_hmm ,correctstates_mazehmmreal] = statesHammingDistance(mazehmmestimate,hmmestimate, testseq);
    
    mazehmmprobs = [mazehmmprobs, correctstates_mazehmm];
    hmmprobs = [hmmprobs, correctstates_hmm];
    mazehmmprobsreal= [mazehmmprobsreal, correctstates_mazehmmreal];
    
end
end

function [correctstates_mazehmm,correctstates_hmm ,correctstates_mazehmmreal] = statesHammingDistance(mazehmmestimate,hmmestimate,testseq)

estimatedstates_mazehmm = mazehmmviterbi(testseq.emissions,testseq.envtype,testseq.rewards,...
    mazehmmestimate.tr_reward,mazehmmestimate.tr_noreward,mazehmmestimate.e_homo,mazehmmestimate.e_hetro);

estimatedstates_hmm = hmmviterbi(testseq.emissions,hmmestimate.tr,hmmestimate.e);

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();
estimatedstates_real=mazehmmviterbi(testseq.emissions,testseq.envtype,testseq.rewards,...
    realTRr,realTRnr,realEhomo,realEhetro);

correctstates_mazehmm = sum(estimatedstates_mazehmm==testseq.states)/200;
correctstates_hmm = sum(estimatedstates_hmm==testseq.states)/200;
correctstates_mazehmmreal = sum(estimatedstates_real==testseq.states)/200;


end

function [mazehmmestimate,hmmestimate] = train_models(seq_data, guess, max_iter) 

[mazehmmestimate.tr_reward, mazehmmestimate.tr_noreward, mazehmmestimate.e_homo, mazehmmestimate.e_hetro] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess.trr ,guess.trnr ,...
    guess.eh, guess.et, 'VERBOSE',false, 'maxiterations', max_iter);


[hmmestimate.tr, hmmestimate.e] = ...
    hmmtrain(seq_data.emissions, guess.trr, guess.eh, 'VERBOSE',false, 'maxiterations', max_iter);

end
