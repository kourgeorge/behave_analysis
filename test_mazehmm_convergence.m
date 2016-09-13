function test_mazehmm_convergence()
% generate a sequence of trials using mazehmmgenerate with two environment type but no rewarded states. 
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters. 
% The guess in this test is the same as the equence generation parametrres.
       

       
test_seq_len();
%test_max_iter();

end

function [tr, e] = get_real_parameters()
eps = 0.05;
e = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps;
             0.5 0.5];
         

tr = [0.8 0.05 0.05 0.05 0.05;
           0.05 0.8 0.05 0.05 0.05;
           0.05 0.05 0.8 0.05 0.05;
           0.05 0.05 0.05 0.8 0.05
           0.2 0.2 0.2 0.2 0.2];

end

function test_seq_len()
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[tr,e] = get_real_parameters();

tr_guess = rand(5);
e_guess = rand(5,2);

env_type_frac = 1;


[envtype,emissions, ~, rewards] = ...
    mazehmmgenerate(10000, tr, tr, ...
    e, e ,env_type_frac, []);

max_iterations = 500;
res = norm(tr_guess-tr) + norm(e-e_guess);
lengths = linspace(10,10000, 15);

for seq_length=lengths
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    res_iter = run_hmm_train(seq_data, tr_guess, tr_guess, e_guess, e_guess, max_iterations);
    res = [res, res_iter(1)+ res_iter(2)];
end

plot([0,lengths], res);
title('Model accuracy vs. sequence length.');
xlabel('# of trials');
ylabel('d(Mreal, Mest.)');
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

[tr, e] = get_real_parameters();

[~, est_trans_noreward, est_emits_homo, ~] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',false, 'maxiterations', max_iterations);

diff_trans = norm(est_trans_noreward - tr);
diff_emits_homo = norm(est_emits_homo - e);
tot_error = [diff_trans, diff_emits_homo];
end

