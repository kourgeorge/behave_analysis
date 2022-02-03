function decodeMazehmmvsHmm()
% generate a sequence of trials using mazehmmgenerate with two environment type but no rewarded states.
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters.
% The guess in this test is the same as the equence generation parametrres.

res_hmmprobs = [];
res_mazehmmprobs = [];
res_mazehmmprobsreal = [];
repetitions = 5;
train_seq_lengths =  floor(linspace(100,1000,5));
for i=1:repetitions
    [hmmprobs, mazehmmprobs, mazehmmprobsreal]=calcseqposteriorprobability(train_seq_lengths);
    res_hmmprobs = [res_hmmprobs; hmmprobs];
    res_mazehmmprobs = [res_mazehmmprobs; mazehmmprobs];
    res_mazehmmprobsreal=[res_mazehmmprobsreal;mazehmmprobsreal];
end

% norm_mazehmmprobs = res_mazehmmprobs-res_mazehmmprobsreal;
% norm_hmmprobs = res_hmmprobs-res_mazehmmprobsreal;

y = [mean(res_mazehmmprobsreal); mean(res_mazehmmprobs); mean(res_hmmprobs)]';

figure
set(gca,'fontsize',22)
hh = bar(train_seq_lengths, y , 'LineWidth',1.5)
legend('SCA-HMM true','SCA-HMM estimated', 'HMM estimated')
xlabel('Sequence length')
ylabel('Posterior -log probability')
title('Comparing the posterior probability of SCA-HMM and HMM')
set(hh(1),'YScale','log')
%ylim([-220, 0])

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

function [hmmprobs, mazehmmprobs, mazehmmprobsreal] = calcseqposteriorprobability(train_seq_lengths)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();


env_type_frac = 0.5;
[envtype,emissions, ~, rewards] = ...
    mazehmmgenerate(1500, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);

[testseq.envtype,testseq.emissions, testseq.states, testseq.rewards] = ...
    mazehmmgenerate(200, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);


guess.tr = createGuessProbabilityMatrices(realTRr, realTRnr, 0.5);
guess.e = createGuessProbabilityMatrices(realEhomo, realEhetro, 0.5);


max_iter = 500;
tolerance = 1e-2;
mazehmmprobs = [];
hmmprobs=[];
mazehmmprobsreal=[];

for seq_length=train_seq_lengths
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    [mazehmmestimate,hmmestimate] = run_train(seq_data, guess, max_iter, tolerance);
    
    postprobs = -decode(mazehmmestimate,hmmestimate, testseq);
    
    mazehmmprobs = [mazehmmprobs, postprobs(1)];
    hmmprobs = [hmmprobs, postprobs(2)];
    mazehmmprobsreal= [mazehmmprobsreal, postprobs(3)];
    
end
end

function postprobs = decode(mazehmmestimate,hmmestimate,testseq)

[~,pSeqmazehmm, ~, ~, ~] = mazehmmdecode(testseq.emissions,testseq.envtype,testseq.rewards,...
    mazehmmestimate.tr_reward,mazehmmestimate.tr_noreward,mazehmmestimate.e_homo,mazehmmestimate.e_hetro);

[~,pSeqhmm, ~, ~, ~] = hmmdecode(testseq.emissions,hmmestimate.tr,hmmestimate.e);

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();
[~,pSeqmazehmmReal, ~, ~, ~]=mazehmmdecode(testseq.emissions,testseq.envtype,testseq.rewards,realTRr,realTRnr,realEhomo,realEhetro);

postprobs = [pSeqmazehmm, pSeqhmm, pSeqmazehmmReal];

end

function [mazehmmestimate,hmmestimate] = run_train(seq_data, guess, max_iter, tolerance) 

[mazehmmestimate.tr_reward, mazehmmestimate.tr_noreward, mazehmmestimate.e_homo, mazehmmestimate.e_hetro] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess.tr ,guess.tr ,...
    guess.e, guess.e, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);


[hmmestimate.tr, hmmestimate.e] = ...
    hmmtrain(seq_data.emissions, guess.tr, guess.e, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);

end


