function res_policies = neurolate_tabular_policies(statesactions)

policies = CreatePolicies(statesactions);

numStates = length(statesactions);
numActions = size(statesactions{1},2);
num_trials = 100;
res_policies = {};
for i=1:length(policies)   
    policy = policies{i};
    states = [];
    actions = [];
    net=patternnet(5);
    net.trainParam.showWindow = false;
    for state=1:numStates
        states = [states;repmat(state, num_trials,1)];
        actions = [actions;(rand(num_trials,1)>policy(state,1))+1];
    end
    
    net = train(net,onehot(states, 1:numStates)',onehot(actions, 1:numActions)');
    res_policies{i}=net;
end


end