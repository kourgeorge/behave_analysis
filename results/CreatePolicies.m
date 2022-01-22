function policies = CreatePolicies(statesactions)

policies = [];
num_states = length(statesactions);
num_policies = size(statesactions{1},1);
for pol=1:num_policies
    policy = [];
    for state=1:num_states
        policy = [policy;statesactions{state}(pol,:)];
    end
    policies=[policies;{policy}];
end
end