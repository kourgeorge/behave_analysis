function policy_divergence = PolicyDivergence(estimated_policies_seq, testseq, theta_hat, theta_gt)

gt_policies = CreatePolicies([{theta_gt.eH},{theta_gt.eT}]);
estimated_policies = CreatePolicies([{theta_hat.eH}, {theta_hat.eT}]);
policy_divergence = zeros(length(testseq.envtype),1);
for t=1:length(testseq.envtype)
    policy_divergence(t)= ComparePolicies(gt_policies{testseq.states(t)}, estimated_policies{estimated_policies_seq(t)});
end
end