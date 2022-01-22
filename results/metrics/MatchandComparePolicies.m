function [match,match_distances, policies_dist] = MatchandComparePolicies(policiesSet1,policiesSet2)
%MATCHPOLICIES Summary of this function goes here
%   Detailed explanation goes here


policies1 = CreatePolicies(policiesSet1);
policies2 = CreatePolicies(policiesSet2);
num_policies = length(policies1);

policies_dist = [];
for i=1:num_policies
    for j=1:num_policies
        policies_dist(i,j) = ComparePolicies(policies1{i},policies2{j});
    end
end

match = matchpairs(policies_dist, 50);
match_distances = [];
for i=1:num_policies
    match_distances(i) = policies_dist(match(i,1), match(i,2));
end

end


