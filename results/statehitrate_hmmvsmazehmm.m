function statehitrate_hmmvsmazehmm()
hitrates = [];
training_seq_lengths = floor(linspace(50,2000,50));
repetitions = 25;

theta_type = 'random';
labels = {'$\hat{\theta}_{HMM}$', '$\hat{\theta}_{CA-HMM}$', '$\theta_{CA-HMM}$'};
%theta_gt = get_real_parameters(0.1);
%theta_gt = getModelParameters(0.1, 'random');


for rep=1:repetitions
        %theta_gt = getModelParameters(0.1, 'gt');
        %theta_gt = get_real_parameters(0.1);
        theta_gt = getGTparameters(4);
        hitrates = cat (3, hitrates, calcHitrate(training_seq_lengths, theta_gt));
end
hitrates = permute(hitrates, [3 1 2]);
show_graph(training_seq_lengths, hitrates, labels, theta_type);
end

function show_graph(x, hitrates, labels ,theta_type)

% hitrates: dim 1 - repetitions
%           dim 2 - sequence length
%           dim 3 - type - hmm, mazehmm, mazehmm_gt

colors = [0 0.4470 0.7410;
    0.8500 0.3250 0.0980;
    0.9290 0.6940 0.1250];


figure;
hold on;
h={};
set(gca,'fontsize',18)
for i=1:size(hitrates,3)
    %errorbar(x,mean(hitrates(:,:,i)),sem(hitrates(:,:,i)), '-s','MarkerSize', 7, 'linewidth', 2, 'CapSize', 16) 
    h(i)={shadedErrorBar(x,hitrates(:,:,i),{@mean,@sem}, {'-','Color',colors(i,:), 'LineWidth',2},3)};
end

legend([h{1}.mainLine,h{2}.mainLine,h{3}.mainLine],[labels(1),labels(2),labels(3)], 'Interpreter','latex')

xlabel('$l^{train}$','Interpreter','latex', 'FontSize', 25)
ylabel('$Pdiv$','Interpreter','latex', 'FontSize', 25)

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


function hitrates = calcHitrate(training_seq_lengths, theta_gt)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.
theta_guess.trR = getrandomdistribution(4,4);
theta_guess.trNR = getrandomdistribution(4,4);
theta_guess.eH = getrandomdistribution(4,2);
theta_guess.eT = getrandomdistribution(4,2);

max_iter = 200;
tolerance = 1e-4;
hitrates = [];

for seq_length=training_seq_lengths
    [trainseq, testseq] = createTrainAndTestSequences(seq_length, 100, theta_gt, 0.5, [0 1; 1 0]);
    %HMM%
    %Train
    [theta_hmm.trR, theta_hmm.eH] = ...
        hmmtrain(trainseq.emissions, theta_guess.trR, theta_guess.eH, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);
    theta_hmm.trNR = theta_hmm.trR;
    theta_hmm.eT = theta_hmm.eH;
    %Estimate
    estimatedstates_hmm = hmmviterbi(testseq.emissions,theta_hmm.trR,theta_hmm.eH);

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
        estimatedstates_mazehmm = ones(1,length(testseq.emissions));
        warning(exp.message);
    end
    %mazeHM using GT
    %Estimate
    estimatedstates_gt = mazehmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
        theta_gt.trR, theta_gt.trNR, theta_gt.eH, theta_gt.eT);
    %%%%

    mazehmm_gt_hitrate= mean(calcPolicyDivergence(estimatedstates_gt, testseq, theta_gt, theta_gt));
    mazehmm_hitrate = mean(calcPolicyDivergence(estimatedstates_mazehmm, testseq, theta_mazehmm,  theta_gt));
    hmm_hitrate = mean(calcPolicyDivergence(estimatedstates_hmm, testseq, theta_hmm, theta_gt));
    hitrates = [hitrates; hmm_hitrate, mazehmm_hitrate, mazehmm_gt_hitrate];
    
end
end

function policy_divergence = calcPolicyDivergence(estimated_policies_seq, testseq, theta_hat, theta_gt)
environment_states = testseq.envtype;
gt_policies = CreatePolicies([{theta_gt.eH},{theta_gt.eT}]);
estimated_policies = CreatePolicies([{theta_hat.eH}, {theta_hat.eT}]);
policy_divergence = [];
for t=1:length(environment_states)
    policy_divergence(t)= ComparePolicies(gt_policies{testseq.states(t)}, estimated_policies{estimated_policies_seq(t)});
end
end

function policy_hit_rate = MatchingPoliciesHitrate(estimated_policies_seq, testseq, theta_hat, theta_gt)
gt_policies = CreatePolicies([{theta_gt.eH},{theta_gt.eT}]);
estimated_policies = CreatePolicies([{theta_hat.eH}, {theta_hat.eT}]);
match = MatchPolicies(gt_policies,estimated_policies);
policy_hit_rate = mean(match(estimated_policies_seq)==testseq.states');
end

function total_policy_dist_sequence = calcActionProbDivergence(estimated_policies_seq, testseq, estimated_theta, true_theta)
environment_states = testseq.envtype;
true_policies_seq = testseq.states;
true_policies = CreatePolicies([{true_theta.eH},{true_theta.eT}]);
estimated_policies = CreatePolicies([{estimated_theta.eH}, {estimated_theta.eT}]);
total_policy_dist_sequence = 0;
sequence_length = length(environment_states);
for t=1:sequence_length
    current_state = environment_states(t);
    current_true_policy = true_policies{true_policies_seq(t)};
    current_estimated_policy = estimated_policies{estimated_policies_seq(t)};
    total_policy_dist_sequence=total_policy_dist_sequence+JSDiv(current_true_policy(current_state,:),current_estimated_policy(current_state,:) );
end
end



function theta_real = get_real_parameters(eps)
theta_real.eH = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

theta_real.eT = [1-eps eps; %o1l2 o2l1
    eps 1-eps;
    eps 1-eps;
    1-eps eps];


theta_real.trR = ...
    [0.7 0.1 0.1 0.1;
    0.1 0.7 0.1 0.1;
    0.1 0.1 0.7 0.1;
    0.1 0.1 0.1 0.7];


theta_real.trNR = ...
    [0.1 0.5 0.2 0.2;
    0.5 0.1 0.2 0.2;
    0.2 0.2 0.1 0.5;
    0.2 0.2 0.5 0.1];

% theta_real.eH = [0.7 0.3;
%     0.5 0.5;
%     0.6 0.4;
%     0.1 0.9];
% 
% theta_real.eT = [0.1 0.9; %o1l2 o2l1
%     0.1 0.9;
%     0.1 0.9;
%     0.1 0.9];

% 
% theta_real.eH = [0.7 0.3;
%     0.5 0.5;
%     0.6 0.4;
%     0.1 0.9];
% 
% theta_real.eT = [0.6 0.4; %o1l2 o2l1
%     0.2 0.8;
%     0.7 0.3;
%     0.3 0.7];



end

