function [next_state, next_action] = predictNextStateandActionByModelParameters(theta, last_state, next_envtype, last_reward)
%gets the model parameters environment type and last reward and returns the
%next action.

if (last_reward==1)
    transition = theta.trR;
else
    transition = theta.trNR;
end

if (next_envtype==1)
    emission = theta.eH;
else
    emission = theta.eT;
end

[~,next_state] = max(transition(last_state,:));
[~,next_action] = max(emission(next_state,:));

end