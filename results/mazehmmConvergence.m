function test_mazehmm_convergence()
% generate a sequence of trials using mazehmmgenerate with two environment type but no rewarded states.
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters.
% The guess in this test is the same as the equence generation parametrres.

tr_res = [];
e_res = [];
steps = 100;
from = 10;
to = 1000;
for i=1:20
    [trdistances,edistances] = getdistanceforsequence(from,to,steps);
    tr_res=[tr_res;trdistances];
    e_res = [e_res; edistances];
end

figure;
set(gca,'fontsize',22)
shadedErrorBar(linspace(from,to,steps),tr_res,{@mean,@std},'*b',3)
xlabel('Sequence length')
%ylabel('KL(E||T)')
ylabel('V(E,T)')
title('Modified Baum Welch - Transition matrices estimation accuracy')

figure();
set(gca,'fontsize',22)
shadedErrorBar(linspace(from,to,steps),e_res,{@mean,@std},'*b',3)
xlabel('Sequence length')
%ylabel('KL(E||T)')
ylabel('V(E,T)')
title('Modified Baum Welch - Emission Matrices Estimation Accuracy')

end

function [trR, trNR, eH, eT] = get_real_parameters()
eps = 0.01;
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

function [Trdistance,Edistance] = getdistanceforsequence(from, to, interval)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();


env_type_frac = 0.5;
[envtype,emissions, ~, rewards] = ...
    mazehmmgenerate(to, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);


tr_guess = getrandomdistribution(4,4);
e_guess = getrandomdistribution(4,2);


max_iter = 500;
Trdistance = [];
Edistance = [];
lengths = linspace(from,to, interval);

for seq_length=floor(lengths)
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    %Ddistanceiter = run_hmm_train(seq_data, tr_guess, tr_guess, e_guess, e_guess, max_iter);
    Ddistanceiter = run_hmm_train(seq_data, realTRr, realTRnr, realEhomo, realEhetro, max_iter);
    Trdistance = [Trdistance, Ddistanceiter(1)];
    Edistance = [Edistance, Ddistanceiter(2)];
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
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',true, 'maxiterations', max_iterations);


%diff_trans = mean([sum(JSDiv(est_trans_reward ,realTRr)), sum(JSDiv(est_trans_noreward ,realTRnr))]);
%diff_emits = mean([sum(JSDiv(est_emits_homo, realEhomo)), sum(JSDiv(est_emits_hetro, realEhetro))]);

diff_trans = mean([sum(sum(abs(est_trans_reward - realTRr))), sum(sum(abs(est_trans_noreward - realTRnr)))]);
diff_emits = mean([sum(sum(abs(est_emits_homo - realEhomo))), sum(sum(abs(est_emits_hetro - realEhetro)))]);


tot_error = [diff_trans, diff_emits];

end

