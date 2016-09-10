function test_hmm_convergance()
% generate a sequence of trials using myhmmgenerate with two environment type but no rewarded states. 
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters. 
% The guess in this test is the same as the equence generation parametrres.

test_seq_len();
test_max_iter();

end


function [guessTRr, guessTRnr, guessEhomo, guessEhetro] = get_guess_parameters()

eps = 0.3;
guessEhomo = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps;
             0.5 0.5];
guessEhetro = [1-eps eps;
             eps 1-eps;
             eps 1-eps;
             1-eps eps;
             0.5 0.5];
         
guessTRr = [0.6 0.1 0.1 0.1 0.1;
           0.1 0.6 0.1 0.1 0.1;
           0.1 0.1 0.6 0.1 0.1;
           0.1 0.1 0.1 0.6 0.1;
           0.1 0.1 0.1 0.1 0.6];
guessTRnr = [0.6 0.1 0.1 0.1 0.1;
           0.1 0.6 0.1 0.1 0.1;
           0.1 0.1 0.6 0.1 0.1;
           0.1 0.1 0.1 0.6 0.1;
           0.1 0.1 0.1 0.1 0.6];

end

function [guessTRr, guessTRnr, guessEhomo, guessEhetro] = get_real_parameters()

eps = 0.05;
guessEhomo = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps;
             0.5 0.5];
guessEhetro = [1-eps eps;
             eps 1-eps;
             eps 1-eps;
             1-eps eps;
             0.5 0.5];
         
guessTRr = [0.8 0.05 0.05 0.05 0.05;
           0.05 0.8 0.05 0.05 0.05;
           0.05 0.05 0.8 0.05 0.05;
           0.05 0.05 0.05 0.8 0.05
           0.2 0.2 0.2 0.2 0.2];
guessTRnr = [0.6 0.25 0.05 0.05 0.05;
           0.25 0.6 0.05 0.05 0.05;
           0.05 0.05 0.6 0.25 0.05;
           0.05 0.05 0.25 0.6 0.05
           0.2 0.2 0.2 0.2 0.2];

end

function test_seq_len()
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[guessTRr, guessTRnr, guessEhomo, guessEhetro] = get_guess_parameters();
[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

env_type_frac = 0.5;

[envtype,emissions, ~, rewards] = ...
    myhmmgenerate(4000, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, []);

max_iterations = 500;
res = [];
lengths = 1000:1000:4000;

for seq_length= lengths
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    res_iter = run_hmm_train(seq_data, guessTRr, guessTRnr, guessEhomo, guessEhetro, max_iterations);
    res = [res; res_iter(1), res_iter(2) + res_iter(2)];
end

plot(lengths, res(:,1));
title('Estimation accuracy of the transition matrix vs. the sequence length.');
xlabel('sequece length (# of trials).');
ylabel('Accuracy measure - dist. between the estimated trans matrix and the real matrix');
end

function test_max_iter()
% checks the correlation between the max numer of iterations and the error in the
% estimated matrices. The larger the max iteration the  more correct should be the trained model.

[guessTRr, guessTRnr, guessEhomo, guessEhetro] = get_guess_parameters();
[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

env_type_frac = 0.5;
[envtype,emissions, ~, rewards] = ...
    myhmmgenerate(1500, realTRr, realTRnr, ...
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
    myhmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',true, 'maxiterations', max_iterations);

diff_trans = norm(est_trans_noreward - realTRnr);
diff_emits_homo = norm(est_emits_homo - realEhomo);
diff_emits_hetro = norm(est_emits_hetro - realEhetro);
tot_error = [diff_trans, diff_emits_homo, diff_emits_hetro];
end

