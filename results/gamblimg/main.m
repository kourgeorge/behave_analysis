p1 = LinearPolicy(1,0);
p2 = LinearPolicy(0.5,-0.2);
trR = [0.9, 0.1; 0.9, 0.1];
trNR = [0.1, 0.9; 0.1, 0.9];
theta_gt.trR = trR;
theta_gt.trNR = trNR;
theta_gt.policies = {p1, p2};

agent = GamblingAgent(theta_gt.trR , theta_gt.trNR, theta_gt.policies);
interaction = GamblingGame(60);
interaction.start(agent)

[interaction.hidden_state;interaction.actions;interaction.wins]


guess_trans_reward = [0.6 0.4; 0.4 0.6];
guess_trans_noreward = [0.8 0.2; 0.2 0.8];

networks = RandomNetworks(2);

[theta_hat.trR, theta_hat.trNR, theta_hat.policies] = ...
    scahmmtrain(interaction.actions, num2cell(interaction.states, 2)' , interaction.wins ...
            ,guess_trans_reward ,guess_trans_noreward , networks);


estimated_policies_seq = scahmmviterbi(interaction.actions, num2cell(interaction.states,2), interaction.rewards,...
    theta_hat.trR, theta_hat.trNR, theta_hat.policies);


%state_hit_rate = mean(estimated_policies_seq==interaction.hidden_state);
%action_hit_rate = mean(policies{estimated_policies_seq}.act()==interaction.actions);

%accuracy = compare_preference_curve(p1 ,NeuralPolicy(models_hat{2}), 100);

[match,match_distances,policies_dist] = MatchGamblingPolicies(theta_gt.policies,NetworksToNeuralPolicies(theta_hat.policies));
