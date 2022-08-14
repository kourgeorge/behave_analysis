
p1 = LinearPolicy(1,0);
p2 = LinearPolicy(1,0.5);

p3 = LinearPolicy(1,0.5);
p4 = LinearPolicy(1,0);


[match,match_distances,policies_dist] = MatchGamblingPolicies({p1,p2}, {p3,p4});

match
