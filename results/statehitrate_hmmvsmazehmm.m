function statehitrate_hmmvsmazehmm()
hitrates = [];
training_seq_lengths = floor(linspace(40,600,20));
repetitions = 20;

theta_type = 'gt2';
labels = {'BW estimated theta', 'MBW estimated theta', 'MBW GT theta'};
theta_gt = getModelParameters(0.01, theta_type);


for seq_length=training_seq_lengths
        hitrates = cat (3, hitrates, calcHitrate(seq_length, repetitions, theta_gt));
end
hitrates = permute(hitrates, [1 3 2]);
show_graph(training_seq_lengths, hitrates, labels, theta_type);
end

function show_graph(x, hitrates, labels ,theta_type)

% hitrates: dim 1 - repetitions
%           dim 2 - sequence length
%           dim 3 - type - hmm, mazehmm, mazehmm_gt

figure;
hold on;
set(gca,'fontsize',22)
for i=1:size(hitrates,3)
    errorbar(x,mean(hitrates(:,:,i)),var(hitrates(:,:,i)), '-s','MarkerSize', 7, 'linewidth', 2, 'CapSize', 16) 
end

xlim([x(1)-10,x(end)+10])
ylim([0.3,1])
legend(labels)
xlabel('Sequence length')
ylabel('Hit-rate')
title(['Hidden state hit-rate - ',theta_type])

hold off;

end

function [trainseq, testseq] = createTrainAndTestSequences(train_len, test_len, theta_gt, env_type_frac, R)
    
    [trainseq.envtype, trainseq.emissions, trainseq.states, trainseq.rewards] = ...
        mazehmmgenerate(train_len, theta_gt.trR, theta_gt.trNR, ...
        theta_gt.eH, theta_gt.eT ,env_type_frac, R);

    [testseq.envtype, testseq.emissions, testseq.states, testseq.rewards] = ...
        mazehmmgenerate(test_len, theta_gt.trR, theta_gt.trNR, ...
        theta_gt.eH, theta_gt.eT ,env_type_frac, R);
    
end


function hitrates = calcHitrate(seq_length, repetitions, theta_gt)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

theta_guess = getModelParameters( 0.01 , 'uniform' );

max_iter = 500;
tolerance = 1e-4;
hitrates = [];

for i=1:repetitions
[trainseq, testseq] = createTrainAndTestSequences(seq_length, 100, theta_gt, 0.5, [1 0; 1 0]);
%HMM%
%Train
[theta_hmm.tr, theta_hmm.e] = ...
    hmmtrain(trainseq.emissions, theta_guess.trR + theta_guess.trNR, theta_guess.eH + theta_guess.eT, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);
%Estimate
estimatedstates_hmm = hmmviterbi(testseq.emissions,theta_hmm.tr,theta_hmm.e);

%MazeHMM%
%Train
[theta_mazehmm.trR, theta_mazehmm.trNR, theta_mazehmm.eH, theta_mazehmm.eT] = ...
    mazehmmtrain(trainseq.emissions, trainseq.envtype , trainseq.rewards ,theta_guess.trR ,theta_guess.trNR ,...
    theta_guess.eH, theta_guess.eT, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);
%Estimate
try
    estimatedstates_mazehmm = mazehmmviterbi(testseq.emissions,testseq.envtype,testseq.rewards,...
        theta_mazehmm.trR,theta_mazehmm.trNR,theta_mazehmm.eH,theta_mazehmm.eT);
catch exp
    estimatedstates_mazehmm = zeros(1,length(testseq.emissions));
    warning(exp.message);
end

%test = mazehmmviterbi(trainseq.emissions,trainseq.envtype,trainseq.rewards,...
%    theta_mazehmm.trR,theta_mazehmm.trNR,theta_mazehmm.eH,theta_mazehmm.eT);

%mazeHM using GT
%Estimate
estimatedstates_real =mazehmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
    theta_gt.trR, theta_gt.trNR, theta_gt.eH, theta_gt.eT);

mazehmm_hitrate = calcCorrectStateRate(estimatedstates_mazehmm, testseq.states);
hmm_hitrate = calcCorrectStateRate(estimatedstates_hmm, testseq.states);
mazehmm_gt_hitrate= calcCorrectStateRate(estimatedstates_real, testseq.states);

hitrates = [hitrates; hmm_hitrate, mazehmm_hitrate, mazehmm_gt_hitrate];

end
end

function corract_state_rate = calcCorrectStateRate(estimated_states, true_states)
corract_state_rate = sum(estimated_states==true_states)/length(true_states);
end

