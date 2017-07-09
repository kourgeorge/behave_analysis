function test_mazehmmperfectguess()

mazehmm_res = [];
hmm_res = [];
steps = 10;
from = 1000;
to = 1000;
repeats = 50;
for i=1:repeats
    [mazeHmmDistance, hmmDistance] = getdistanceforsequence();
    mazehmm_res=[mazehmm_res; mazeHmmDistance];
    hmm_res = [hmm_res; hmmDistance];
    
end
mean(mazehmm_res)
std(mazehmm_res)

mean(hmm_res)
std(hmm_res)

end

function [trR, trNR, eH, eT] = get_real_parameters()
eH = [0.1 0.9;
    0.9 0.1];

eT = [0.9 0.1;
    0.1 0.9];

trR = [0.9 0.1
    0.1 0.9];

trNR = [ 0 0; 0 0];
end

function [mazeHmmDistance, hmmDistance] = getdistanceforsequence()

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

env_type_frac = 0.5;
sequenceLength = 1000;

[envtype,emissions, ~, rewards] = ...
    mazehmmgenerate(sequenceLength, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 1; 1 1]);

max_iter = 500;
tolerance = 1e-4;
mazeHmmDistance = [];
hmmDistance = [];

seq_data.envtype = envtype;
seq_data.emissions = emissions;
seq_data.rewards = rewards;

guessTRr = [ .5 .5; .5 .5];

[mazeHmmDistance, hmmDistance] = run_hmm_train(seq_data, guessTRr, guessTRr, realEhomo, realEhetro, max_iter, tolerance);

end


function [mazeHmmDistance, hmmDistance] = run_hmm_train(seq_data, guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro, max_iterations, tolerance)

[realTRr, realTRnr, realEhomo, realEhetro] = get_real_parameters();

[~, ~, est_emits_homo, est_emits_hetro, ~] = ...
    mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',true, 'maxiterations', max_iterations, 'TOLERANCE', tolerance);


[~, est_emits] = ...
    hmmtrain(seq_data.emissions, guess_trans_noreward, guess_emit_homo, 'VERBOSE',true, 'maxiterations', max_iterations, 'TOLERANCE', tolerance);

mazeHmmDistance = mean([sum(sum(abs(est_emits_homo - realEhomo))) , sum(sum(abs(est_emits_hetro - realEhetro)))]);
hmmDistance = mean([sum(sum(abs(est_emits - realEhomo))), sum(sum(abs(est_emits - realEhetro)))]);

end


