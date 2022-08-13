function [trans_JS,policies_JS] = paramatersJS (theta_gt,theta_hat)
%PARAMATERSJS Summary of this function goes here
%   Detailed explanation goes here

if isfield(theta_gt, 'eH')
    [match, match_distances, policies_dist] = MatchandComparePolicies([{theta_gt.eH},{theta_gt.eT}],[{theta_hat.eH}, {theta_hat.eT}]);
else
    [match,match_distances,policies_dist] = MatchGamblingPolicies(theta_gt.policies, theta_hat.policies);
end

match= match(:,1);
est_trans_noreward = theta_hat.trNR(match,:);
est_trans_noreward = est_trans_noreward(:,match);
est_trans_reward = theta_hat.trR(match,:);
est_trans_reward = est_trans_reward(:,match);

trans_JS = mean([mean(JSDiv(est_trans_reward ,theta_gt.trR)), mean(JSDiv(est_trans_noreward ,theta_gt.trNR))]);
policies_JS = mean(match_distances);

end

