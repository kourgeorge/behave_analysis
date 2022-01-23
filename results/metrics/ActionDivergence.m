function total_policy_dist_sequence = ActionDivergence(estimated_policies_seq, testseq, estimated_theta, true_theta)
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

