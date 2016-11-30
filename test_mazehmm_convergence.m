function test_mazehmm_convergence()
% generate a sequence of trials using mazehmmgenerate with two environment type but no rewarded states.
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters.
% The guess in this test is the same as the equence generation parametrres.

res = [];
for i=1:50
    res=[res;getdistanceforsequence(10,1000,30)];
end

mean_res = mean(res);

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

function Ddistance = getdistanceforsequence(from, to, interval)
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
Ddistance = [];
lengths = linspace(from,to, interval);

for seq_length=floor(lengths)
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    Ddistanceiter = run_hmm_train(seq_data, tr_guess, tr_guess, e_guess, e_guess, max_iter);
    Ddistance = [Ddistance, Ddistanceiter(1)];
end
end




function test_max_iter()
% checks the correlation between the max numer of iterations and the error in the
% estimated matrices. The larger the max iteration the  more correct should be the trained model.

[guessTRr, guessTRnr, guessEhomo, guessEhetro] = get_guess_parameters();
[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

env_type_frac = 0.5;
[envtype,emissions, ~, rewards] = ...
    mazehmmgenerate(1500, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, []);

max_iters = 1000:1000:4000;

res = [];
for max_iter = max_iters
    
    seq_data.envtype = envtype;
    seq_data.emissions = emissions;
    seq_data.rewards = rewards;
    res_iter = run_hmm_train(seq_data, guessTRr, guessTRnr, guessEhomo, guessEhetro, max_iter);
    res = [res; res_iter(1), res_iter(2) + res_iter(3)];
end

plot(max_iters, res(:,1));
title('Estimation accuracy of the transition matrix vs. max number of Baum Welch iterations.');
xlabel('max iteration');
ylabel('accuracy (dist. between the estimated and real trans matrix)');
end

function tot_error = run_hmm_train(seq_data, guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro, max_iterations)

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',false, 'maxiterations', max_iterations);

diff_trans = sum(KLDiv(est_trans_reward ,realTRr));
diff_emits = sum(KLDiv(est_emits_homo, realEhomo));
tot_error = [diff_trans, diff_emits];

end

