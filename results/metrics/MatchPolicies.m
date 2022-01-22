function match = MatchPolicies(policySet1,policySet2)
%MATCHPOLICIES Summary of this function goes here
%   Detailed explanation goes here
num_policies = length(policySet1);

policies_dist = [];
for i=1:num_policies
    for j=1:num_policies
        policies_dist(i,j) = ComparePolicies(policySet1{i},policySet2{j});
    end
end

match = matchpairs(policies_dist, 50);
match = match(:,1);

end

