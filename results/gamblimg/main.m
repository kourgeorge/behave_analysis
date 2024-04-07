close all;
p1 = LinearPolicy(1,0);
p2 = AntiLinearPolicy(1,0);
% % 
% p1 = BandPolicy(1,0,0.3);
% p2 = BandPolicy(-1,0.3,0.6);


theta_gt= get_neural_gt_parameters();
%theta_gt.policies = {p1, p2};



agent = GamblingAgent(theta_gt.trR , theta_gt.trNR, theta_gt.policies);
interaction = GamblingGame(1000);
interaction.start(agent)
%interaction.draw_empirical_decision_boundary()

[interaction.hidden_states;interaction.actions;interaction.wins]

%create a network mimiking the band policy
 %mimicking_network = NetworksToNeuralPolicies(MimicGamblingPolicy({p1}, 100));
 %CompareGamblingPolicies(mimicking_network{1} ,theta_gt.policies{1}, 500);
 
 
guess = get_neural_random_gt_parameters();
guess.policies = RandomNetworks(2);


[theta_hat.trR, theta_hat.trNR, theta_hat.policies] = ...
    scahmmtrain(interaction.actions, num2cell(interaction.states, 2)' , interaction.wins ...
            ,guess.trR ,guess.trNR , guess.policies);

% compare the policies in each set of parameters.
distance_gts = CompareGamblingPolicies(theta_gt.policies{1} ,theta_gt.policies{2}, 500);

guess_policies = NetworksToNeuralPolicies(guess.policies);
distance_guesses = CompareGamblingPolicies(guess_policies{1} ,guess_policies{2}, 500);

hat_policies = NetworksToNeuralPolicies(theta_hat.policies);
distance_estimated = CompareGamblingPolicies(hat_policies{1} ,hat_policies{2}, 500);


[gues_match,guess_match_distances,guess_policies_dist] = ...
    MatchGamblingPolicies(theta_gt.policies, NetworksToNeuralPolicies(guess.policies));
[match,match_distances,policies_dist] = ...
    MatchGamblingPolicies(theta_gt.policies,NetworksToNeuralPolicies(theta_hat.policies));

sum(guess_match_distances)
sum(match_distances)


estimatedstatesNBIRD = scahmmviterbi(interaction.actions, num2cell(interaction.states,2), interaction.rewards,...
    theta_hat.trR, theta_hat.trNR, theta_hat.policies);

estimatedstatesNBIRDGuess = scahmmviterbi(interaction.actions, num2cell(interaction.states,2), interaction.rewards,...
    guess.trR, guess.trNR, guess.policies);

policy_hit_rate = MatchingCBIRDPoliciesHitrate(estimatedstatesNBIRD, interaction.hidden_states, theta_hat, theta_gt);


%Pdiv
% 
% table_policies = tabulate_neural_policies(theta_hat.policies, 2);
% theta_hat.eH = table_policies{1};
% theta_hat.eT = table_policies{2};
% 
% table_policies = tabulate_neural_policies(theta_gt.policies, 2);
% theta_gt.eH = table_policies{1};
% theta_gt.eT = table_policies{2};
%     
%     
% NBIRDPdiv = mean(PolicyDivergence(estimatedstatesNBIRD, interaction.hidden_states, theta_hat, theta_gt));
% NBIRDPdivGuess = mean(PolicyDivergence(estimatedstatesNBIRDGuess, interaction.hidden_states, guess, theta_gt));
% 
