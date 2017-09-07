function [next_state, next_action] = predictNextStateandActionByModelParameters(estimated_parameters, last_state, next_envtype, last_reward)
%gets the model parameters environment type and last reward and returns the
%next action.

if (last_reward)
    transition = estimated_parameters.est_trans_reward;
else
    transition = estimated_parameters.est_trans_noreward;
end

if (next_envtype)
    emission = estimated_parameters.est_emits_homo;
else
    emission = estimated_parameters.est_emits_hetro;
end

[~,next_state] = max(transition(last_state,:));
[~,next_action] = max(emission(next_state,:));

end