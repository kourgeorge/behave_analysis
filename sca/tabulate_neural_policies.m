function state_action_probs = tabulate_neural_policies(policies, numStates)
%MODEL_TO_TABULAR Converts neural models with discrete state and action
%space to tabular representation, each table contains the 

numPolicies = length(policies);
state_action_probs = cell(numStates,1);

all_states = eye(numStates);

% precalculate all states in a batch.
policy_state_action = cell(numPolicies,1);
for policy=1:numPolicies
      policy_state_action{policy} = policies{policy}(all_states')';
end
    
%order the datastructure is state oriented fashion.
for state=1:numStates
    for policy=1:numPolicies
        state_action_probs{state} = [state_action_probs{state};policy_state_action{policy}(state,:)];
    end
end

end

