function policy_hit_rate = MatchingPoliciesHitrate(estimated_policies_seq, testseq, theta_hat, theta_gt)
gt_policies = CreatePolicies([{theta_gt.eH},{theta_gt.eT}]);
estimated_policies = CreatePolicies([{theta_hat.eH}, {theta_hat.eT}]);
match = MatchPolicies(gt_policies,estimated_policies);
policy_hit_rate = match(estimated_policies_seq)==testseq.states';
end