function [ output_args ] = mazehmmPredictionAbility()
%MAZEHMMPREDICTIONABILITY Summary of this function goes here
%   Detailed explanation goes here

%The goal of this test is to be comparable to the action prediction test
%that is perfomed on the real word data.
%In that test we try to predict the next action based on the obaservations
%of the last sequence of actions. The training is done on the same
%sequence. However here since the data is synthetic, the ground truth is
%available. Nevertheless, we will use the exact same methodology olso for
%the the curresnt state estimation.


%The following is the descrition of this test flow.
%Create observations sequence given the true model.
%Estimate the model parameters or use random parameters.(*)
%Predict the next action
%Calculate the accuracy of predicting underlying strategy at time $n$ given
%the true undelying strategy at time $n-1$ and action at at trial $n$

%(*) Estimating the model parameters is done in several ways:
%1. random transition matrices (but true emmission matrices)
%2. estimation based on shuffled sequence
%3. estimation based on the training sequence

%Dimensions: 1. different training lengths. 2.different true model parameters.
%3. Drifting model parameters??? We still not sure how to do this.

[data.envtype,data.actions,~,data.rewards] = generate_synthetic_sequence(2500);

sequence_lengths = 50:50:700;
repetitions  = 100;
action_prediction_accuracy_per_length = [];
state_prediction_accuracy_per_length = [];
for sequence_length = sequence_lengths
    action_prediction_accuracy_inorder = [];
    action_prediction_accuracy_shuffled = [];
    action_prediction_accuracy_gt = [];
    action_prediction_accuracy_random = [];
    
    state_prediction_accuracy_inorder = [];
    state_prediction_accuracy_shuffled = [];
    state_prediction_accuracy_gt = [];
    state_prediction_accuracy_random = [];
    
    for i=1:repetitions
        
        start_train_ind = randi([1,1500-sequence_length]);
        end_train_ind = start_train_ind+sequence_length;
        train_envtype = data.envtype(start_train_ind:end_train_ind);
        train_actions = data.actions(start_train_ind:end_train_ind);
        train_rewards = data.rewards(start_train_ind:end_train_ind);
        
        theta = estimate_model_parameters(train_envtype,train_actions,train_rewards);
        
        permutation = randperm(length(train_actions));
        theta_s = estimate_model_parameters(train_envtype(permutation),train_actions(permutation),train_rewards(permutation));

        [theta_gt.est_trans_noreward,...
            theta_gt.est_trans_reward,...
            theta_gt.est_emits_homo,...
            theta_gt.est_emits_hetro] = getModelParameters( 0.01 , 'guess' );
        
        [theta_r.est_trans_noreward,...
            theta_r.est_trans_reward,...
            theta_r.est_emits_homo,...
            theta_r.est_emits_hetro] = getModelParameters( 0.01 , 'random' );
        
        [test_envtype,test_actions,test_states,test_rewards]= generate_synthetic_sequence(100);
   
        [action_accuracy, state_accuracy] = next_action_state_prediction_accuracy(test_envtype,test_actions, test_states, test_rewards, theta);
        [s_action_accuracy, s_state_accuracy] = next_action_state_prediction_accuracy(test_envtype,test_actions, test_states, test_rewards, theta_s);
        [gt_action_accuracy, gt_state_accuracy] = next_action_state_prediction_accuracy(test_envtype,test_actions, test_states, test_rewards, theta_gt);
        [r_action_accuracy, r_state_accuracy] = next_action_state_prediction_accuracy(test_envtype,test_actions, test_states, test_rewards, theta_r);
        
        action_prediction_accuracy_inorder = [action_prediction_accuracy_inorder, action_accuracy];
        action_prediction_accuracy_shuffled = [action_prediction_accuracy_shuffled, s_action_accuracy];
        action_prediction_accuracy_gt = [action_prediction_accuracy_gt, gt_action_accuracy];
        action_prediction_accuracy_random = [action_prediction_accuracy_random, r_action_accuracy];
        
        state_prediction_accuracy_inorder = [state_prediction_accuracy_inorder, state_accuracy];
        state_prediction_accuracy_shuffled = [state_prediction_accuracy_shuffled, s_state_accuracy];
        state_prediction_accuracy_gt = [state_prediction_accuracy_gt, gt_state_accuracy];
        state_prediction_accuracy_random = [state_prediction_accuracy_random, r_state_accuracy];
    end
    action_prediction_accuracy_per_length = [action_prediction_accuracy_per_length;...
        mean(action_prediction_accuracy_random), mean(action_prediction_accuracy_shuffled), [mean(action_prediction_accuracy_inorder), mean(action_prediction_accuracy_gt)]];
    state_prediction_accuracy_per_length = [state_prediction_accuracy_per_length;...
        mean(state_prediction_accuracy_random), mean(state_prediction_accuracy_shuffled), [mean(state_prediction_accuracy_inorder), mean(state_prediction_accuracy_gt)]];

end

subplot(1,2,1)
plot(sequence_lengths, action_prediction_accuracy_per_length)
legend('random', 'shuffled', 'inorder', 'true parameters' )
ylim([0,1])
xlabel('Sequence length')
ylabel('Action prediction accuracy')

subplot(1,2,2)
plot(sequence_lengths, state_prediction_accuracy_per_length)
legend('random', 'shuffled', 'inorder', 'true parameters' )
ylim([0,1])
xlabel('Sequence length')
ylabel('State prediction accuracy')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Helper Functions %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function estimated_parameters = estimate_model_parameters(envtype,action,reward)
eps = 0.01;
[ guess_trans_noreward,  guess_trans_reward, guess_emit_homo, guess_emit_hetro] = getModelParameters( eps, 'uniform');
[estimated_parameters.est_trans_reward, ...
    estimated_parameters.est_trans_noreward, ...
    estimated_parameters.est_emits_homo, ...
    estimated_parameters.est_emits_hetro] = mazehmmtrain(action, envtype, reward, ...
    guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro,...
    'VERBOSE', false, 'maxiterations', 500, 'TOLERANCE',0.01);
end



function [envtype,seq,states,rewards] = generate_synthetic_sequence(num_trials)

[ trNR,  trR, eH, eT] = getModelParameters( 0.01 , 'guess' );
[envtype,seq,states,rewards] = mazehmmgenerate(num_trials, trR, trNR, eH, eT, 0.5, [1 0; 1 0] );

end

function estimatedstates = estimate_hidden_states(actions,envtype,rewards,theta)
estimatedstates = mazehmmviterbi(actions,envtype,rewards, ...
    theta.est_trans_reward, theta.est_trans_noreward, ... 
    theta.est_emits_homo, theta.est_emits_hetro);
end

function [action_accuracy, state_accuracy] = next_action_state_prediction_accuracy(test_envtype,test_actions, test_states,test_rewards, theta)

estimated_states = estimate_hidden_states(test_actions(1:end-1), test_envtype(1:end-1), test_rewards(1:end-1), theta);

last_state = estimated_states(end);
last_reward = test_rewards(end-1);
next_envtype = test_rewards(end);
next_action_actual = test_actions(end);
next_state_actual = test_states(end);

[next_state_prediction, next_action_prediction] = ...
    predictNextStateandActionByModelParameters(theta, last_state, next_envtype, last_reward);

action_accuracy = next_action_actual ==next_action_prediction;
state_accuracy = next_state_actual == next_state_prediction;
end