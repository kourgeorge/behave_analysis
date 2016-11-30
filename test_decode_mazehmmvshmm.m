function test_decode_mazehmmvshmm()
% generate a sequence of trials using mazehmmgenerate with two environment type but no rewarded states.
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters.
% The guess in this test is the same as the equence generation parametrres.

res_hmmprobs = [];
res_mazehmmprobs = [];
res_mazehmmprobsreal = [];
for i=1:2
    [hmmprobs, mazehmmprobs, mazehmmprobsreal]=calcseqposteriorprobability(100,1000,5);
    res_hmmprobs = [res_hmmprobs; hmmprobs];
    res_mazehmmprobs = [res_mazehmmprobs; mazehmmprobs];
    res_mazehmmprobsreal=[res_mazehmmprobsreal;mazehmmprobsreal];
end

norm_mazehmmprobs = res_mazehmmprobs-res_mazehmmprobsreal;
norm_hmmprobs = res_hmmprobs-res_mazehmmprobsreal;


bar(floor(linspace(100,1000,5))', [mean(res_mazehmmprobsreal); mean(res_mazehmmprobs); mean(res_hmmprobs)]','LineWidth',1.5)
legend('sca-hmm real','sca-hmm', 'hmm')

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


guess.tr = getrandomdistribution(4,4);
guess.e = getrandomdistribution(4,2);


max_iter = 1500;
mazehmmprobs = [];
hmmprobs=[];
mazehmmprobsreal=[];
lengths = linspace(from,to, interval);

for seq_length=floor(lengths)
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    [mazehmmestimate,hmmestimate] = run_train(seq_data, guess, max_iter);
    
    postprobs = decode(mazehmmestimate,hmmestimate, testseq);
    
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

function [mazehmmestimate,hmmestimate] = run_train(seq_data, guess, max_iter) 

[mazehmmestimate.tr_reward, mazehmmestimate.tr_noreward, mazehmmestimate.e_homo, mazehmmestimate.e_hetro] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess.tr ,guess.tr ,...
    guess.e, guess.e, 'VERBOSE',false, 'maxiterations', max_iter);


[hmmestimate.tr, hmmestimate.e] = ...
    hmmtrain(seq_data.emissions, guess.tr, guess.e, 'VERBOSE',false, 'maxiterations', max_iter);

end


