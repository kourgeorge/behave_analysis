function Adiv = ActionDivergence(estimated_policies_seq, testseq, estimated_theta, true_theta)
true_policies = CreatePolicies([{true_theta.eH},{true_theta.eT}]);
estimated_policies = CreatePolicies([{estimated_theta.eH}, {estimated_theta.eT}]);

Adiv = [];
sequence_length = length(testseq.envtype);
for t=1:sequence_length
    current_state = testseq.envtype(t);
    current_true_policy = true_policies{testseq.states(t)};
    current_estimated_policy = estimated_policies{estimated_policies_seq(t)};
    Adiv(t)=JSDiv(current_true_policy(current_state,:),current_estimated_policy(current_state,:) );
end
end

