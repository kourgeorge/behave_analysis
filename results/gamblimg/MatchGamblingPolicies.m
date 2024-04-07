function [match,match_distances,policies_dist] = MatchGamblingPolicies(policies_set1, policies_set2)
%MATCHGAMBLINGPOLICIES Summary of this function goes here
%   Detailed explanation goes here


num_policies = length(policies_set1);


policies_dist = [];
for i=1:num_policies
    for j=1:num_policies
        policies_dist(i,j) = CompareGamblingPolicies(policies_set1{i},policies_set2{j}, 250);
    end
end

match = matchpairs(policies_dist, 50);
match_distances = [];
for i=1:num_policies
    match_distances(i) = policies_dist(match(i,1), match(i,2));
end

end


