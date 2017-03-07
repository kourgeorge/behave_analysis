function test_mazehmmvshmm()
% generate a sequence of trials using mazehmmgenerate with two environment type but no rewarded states.
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters.
% The guess in this test is the same as the equence generation parametrres.

res_hmm = [];
res_mazehmm = [];
for i=1:50
    [hmmdistance, mazehmmdistance]=getdistanceforsequence(100,1000,5);
    res_hmm = [res_hmm; hmmdistance];
    res_mazehmm = [res_mazehmm; mazehmmdistance];
end

mean_res1 = mean(res_hmm);
mean_res2 = mean(res_mazehmm);

plot(mean_res);
title('Model accuracy vs. sequence length.');
xlabel('# of trials');
ylabel('d(Mreal, Mest.)');


%test_max_iter();

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

function [hmmdistance, mazehmmdistance] = getdistanceforsequence(from, to, interval)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();


env_type_frac = 0.5;
[envtype,emissions, ~, rewards] = ...
    mazehmmgenerate(1500, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 1 0]);


tr_guess = getrandomdistribution(4,4);
e_guess = getrandomdistribution(4,2);


max_iter = 1500;
mazehmmdistance = [];
hmmdistance=[];
lengths = linspace(from,to, interval);

for seq_length=floor(lengths)
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    Ddistanceiter = run_hmm_train(seq_data, tr_guess, tr_guess, e_guess, e_guess, max_iter);
    mazehmmdistance = [mazehmmdistance, Ddistanceiter(1)];
    hmmdistance = [hmmdistance, Ddistanceiter(2)];
    
end
end


function tot_error = run_hmm_train(seq_data, guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro, max_iterations)

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();
[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',false, 'maxiterations', max_iterations);


[est_trans_hmm, est_emits] = ...
    hmmtrain(seq_data.emissions, guess_trans_noreward, guess_emit_homo, 'VERBOSE',false, 'maxiterations', max_iterations);

diff_trans = sum(KLDiv(est_trans_reward ,realTRr)) + sum(KLDiv(est_trans_noreward ,realTRnr));
diff_trans_hmm = sum(KLDiv(est_trans_hmm, realTRr)) + sum(KLDiv(est_trans_hmm ,realTRnr));
tot_error = [diff_trans, diff_trans_hmm];

end

