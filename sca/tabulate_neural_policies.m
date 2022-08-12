function state_action_probs = tabulate_neural_policies(policies, numStates)
%MODEL_TO_TABULAR Converts neural models with discrete state and action
%space to tabular representation, each table contains the 

numPolicies = length(policies);
state_action_probs = cell(numStates,1);

for state=1:numStates
    state_onehot = onehot(state, 1:numStates);
    for policy=1:numPolicies
        state_action_probs{state} = [state_action_probs{state};policies{policy}(state_onehot')'];
    end
end

end

