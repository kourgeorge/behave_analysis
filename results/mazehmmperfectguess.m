function test_mazehmmperfectguess()

tr_res = [];
e_res = [];
iterations_res = [];
steps = 10;
from = 0;
to = 1;
repeats = 50;
for i=1:repeats
    [trdistances,edistances, numIterations] = getdistanceforsequence(from,to,steps);
    tr_res=[tr_res; trdistances];
    e_res = [e_res; edistances];
    iterations_res = [iterations_res; numIterations];
    
end

figure;
set(gca,'fontsize',22)
bar(linspace(from,to,steps), mean(tr_res))
hold on;
errorb(linspace(from,to,steps), mean(tr_res),std(tr_res), 'linewidth', 1)
xlabel('Noise Factor')
%ylabel('KL(E||T)')
ylabel('V(E,T)')
title('Estimated transition probilities accuracy')
hold off;

figure;
set(gca,'fontsize',22)
bar(linspace(from,to,steps), mean(e_res))
hold on;
errorb(linspace(from,to,steps), mean(e_res),std(e_res), 'linewidth', 1)
xlabel('Noise Factor')
ylabel('V(E,T)')
title('Estimation Emission matrices accuracy')
hold off;


figure;
set(gca,'fontsize',22)
bar(linspace(from,to,steps), mean(iterations_res))
hold on;
errorb(linspace(from,to,steps), mean(iterations_res),std(iterations_res), 'linewidth', 1)
xlabel('Noise Factor')
ylabel('Number of iterations')
title('Iteration needed for convergence')
hold off

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

function [Trdistance, Edistance, numIterations] = getdistanceforsequence(noiseFrom, noiseTo, interval)

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

env_type_frac = 0.5;
sequenceLength = 1000;

[envtype,emissions, ~, rewards] = ...
    mazehmmgenerate(sequenceLength, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);

max_iter = 500;
tolerance = 1e-4;
Trdistance = [];
Edistance = [];
numIterations = [];
noiseVec = linspace(noiseFrom, noiseTo, interval);

for noise=noiseVec
    
    seq_data.envtype = envtype;
    seq_data.emissions = emissions;
    seq_data.rewards = rewards;
    
    [guessTRr, guessTRnr] = createGuessTransitionsParameters(realTRr, realTRnr, noise) ;
    
    Ddistanceiter = run_hmm_train(seq_data, guessTRr, guessTRnr, realEhomo, realEhetro, max_iter, tolerance);
    Trdistance = [Trdistance, Ddistanceiter(1)];
    Edistance = [Edistance, Ddistanceiter(2)];
    numIterations = [numIterations, Ddistanceiter(3)];
end
end


function tot_error = run_hmm_train(seq_data, guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro, max_iterations, tolerance)

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro, logliks] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',true, 'maxiterations', max_iterations, 'TOLERANCE', tolerance);

diff_trans = mean([sum(sum(abs(est_trans_reward - realTRr))), sum(sum(abs(est_trans_noreward - realTRnr)))]);
diff_emits = mean([sum(sum(abs(est_emits_homo - realEhomo))), sum(sum(abs(est_emits_hetro - realEhetro)))]);

numIterations = length(logliks);

tot_error = [diff_trans, diff_emits, numIterations];

end


