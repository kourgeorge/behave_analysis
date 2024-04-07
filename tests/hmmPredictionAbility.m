function [ output_args ] = hmmPredictionAbility()
%MAZEHMMPREDICTIONABILITY Summary of this function goes here
%   Detailed explanation goes here

%The goal of this test is to compare the prediction performance of HMM to
%the performance if SCA-HMM on the following experimental paradigm. 

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

[data.actions,~] = generate_synthetic_sequence(2500);

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
        train_actions = data.actions(start_train_ind:end_train_ind);
        
        theta = estimate_model_parameters(train_actions);
        
        permutation = randperm(length(train_actions));
        theta_s = estimate_model_parameters(train_actions(permutation));

        theta_gt = getModelParameters( 0.01 , 'guess' );
        
        theta_r = getModelParameters( 0.01 , 'random' );
        
        [test_actions,test_states]= generate_synthetic_sequence(100);
        
        [action_accuracy, state_accuracy] = next_action_state_prediction_accuracy(test_actions, test_states, theta);
        [s_action_accuracy, s_state_accuracy] = next_action_state_prediction_accuracy(test_actions, test_states, theta_s);
        [gt_action_accuracy, gt_state_accuracy] = next_action_state_prediction_accuracy(test_actions, test_states, theta_gt);
        [r_action_accuracy, r_state_accuracy] = next_action_state_prediction_accuracy(test_actions, test_states,  theta_r);
        
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

function estimated_parameters = estimate_model_parameters(action)
eps = 0.01;
theta = getModelParameters( eps, 'uniform');
[estimated_parameters.tr, ...
    estimated_parameters.em] = hmmtrain(action, theta.trR, theta.eH,...
    'VERBOSE', false, 'maxiterations', 500, 'TOLERANCE',0.01);
end



function [seq,states] = generate_synthetic_sequence(num_trials)

theta = getModelParameters( 0.01 , 'guess' );
[seq,states] = hmmgenerate(num_trials, theta.trR, theta.eH);

end

function estimatedstates = estimate_hidden_states(actions, theta)
estimatedstates = hmmviterbi(actions, theta.tr, theta.em);
end

function [action_accuracy, state_accuracy] = next_action_state_prediction_accuracy(test_actions, test_states, theta)

estimated_states = estimate_hidden_states(test_actions(1:end-1), theta);

last_state = estimated_states(end);
next_action_actual = test_actions(end);
next_state_actual = test_states(end);

[~,next_state_prediction] = max(theta.tr(last_state,:));
[~,next_action_prediction] = max(theta.em(next_state_prediction,:));

action_accuracy = next_action_actual ==next_action_prediction;
state_accuracy = next_state_actual == next_state_prediction;
end