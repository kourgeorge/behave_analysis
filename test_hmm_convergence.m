function test_hmm_convergence()
% generate a sequence of trials using mazehmmgenerate with two environment type but no rewarded states.
% The estimated transition and emmits probabilities should be the same as the sequence generation parameters.
% The guess in this test is the same as the equence generation parametrres.


res = [];
steps = 5;
for i=1:5
    res=[res;getdistanceforsequence(10,1000,5)];
end

f=fit(linspace(10,1000,steps)',mean(res)','poly2');
shadedErrorBar(linspace(1,1000,steps),res,{@mean,@std},'*b',3)
hold on
plot(f)
xlabel('Sequnce length')
ylabel('KL(E||T)')
title('Baum Welch - Estimation accuracy vs. Sequence length')
hold off

%test_max_iter();

end

function [tr, e] = get_real_parameters()
eps = 0.05;
e = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];


tr = [0.7 0.1 0.1 0.1;
    0.1 0.7 0.1 0.1;
    0.1 0.1 0.7 0.1;
    0.1 0.1 0.1 0.7];

end

function Ddistance = getdistanceforsequence(from, to, interval)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[tr,e] = get_real_parameters();
tr_guess = getrandomdistribution(4,4);
e_guess = getrandomdistribution(4,2);

emissions = hmmgenerate(floor(1000), tr, e);

max_iterations = 1500;
Ddistance = [];
lengths = linspace(from,to, interval);

for seq_length=lengths
    Ddistanceiter = run_hmm_train(emissions(1:floor(seq_length)), tr_guess, e_guess, max_iterations);
    Ddistance = [Ddistance, Ddistanceiter(1)];
end
end


function tot_error = run_hmm_train(emissions, guess_trans, guess_emit, max_iterations)

[tr, e] = get_real_parameters();

[est_trans,est_emits,~]  = hmmtrain(emissions, guess_trans, guess_emit, 'VERBOSE',false, 'maxiterations', max_iterations);

diff_trans = sum(KLDiv(est_trans ,tr));
diff_emits = sum(KLDiv(est_emits, e));
tot_error = [diff_trans, diff_emits];
end

