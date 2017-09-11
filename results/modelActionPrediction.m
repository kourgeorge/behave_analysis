function modelActionPrediction()
%MODELACTIONPREDICTION Calculates the next step prediction accuracy based
%on a training set that is used to calculate the model parameters. The goal
%validate the model. We run the model on synthetic data created based on
%model parameters. We want to show that when the data is created from the
%presumed model, we are able to estimate the model parameters.
%Thus, we do the same however in this case we shufffle the damples, which
%keep the same statistics however looses the continuity of the process.
%In this case we assume that the estimation process will fail to estimate
%the parameters.

%Given a sequence, estimate the model parameters
%Use a subsequence and estimate the last state using the viterbi algorithm
%Predict the state of the next state.
%Perform for different sequences size, and for each make several
%repititions.
%(*)Then we do the same but before we shuffle the sequence of observations.

%We can now measure the following:
%1. The accuracy of estimating a state.
%2. Given a state, measure the accuracy of the next state.


sequence_length_acc = [];
scrambled_sequence_length_acc = [];
seq_lengthes = 50:50:500;
for i=1:seq_lengthes
    
    %create a synthetic data using the sca-hmm data generator
    [envtype,seq,states,rewards] = generate_synthetic_sequence(1500);
    
    % convert the synthetic data to the form we get from the actual data
    behave_data = convert_synthetic_data_to_behavior_data(envtype,seq,states,rewards);
    
    % run experiment of on each generated sequence
    [accuracy_per_sequence, ~] = runExperimentOnSequence(seq_lengthes, behave_data);
    sequence_length_acc = [sequence_length_acc; accuracy_per_sequence];

    % now scramble the sequence
    % create a permutation 
    p = randperm(length(seq));
    scrambled_behave_data = behave_data(p,:);
    
    % run experiment of on each generated sequence
    [accuracy_per_scrambled_sequence, ~] = runExperimentOnRat(seq_lengthes, scrambled_behave_data);
    scrambled_sequence_length_acc = [scrambled_sequence_length_acc; accuracy_per_scrambled_sequence];
    
    plotresult(i, seq_lengthes, [accuracy_per_scrambled_sequence', accuracy_per_sequence']);
    title(['Seq ', num2str(i)])
    
end
plotresult(8, seq_lengthes, [nanmean(scrambled_sequence_length_acc)', nanmean(sequence_length_acc)']);
title('Average Accuracy')
suptitle('Next step prediction accuracy vs. training sequence length.')

end

function plotresult(sub, seq_lengthes, data)

subplot(2,4,sub)
plot(seq_lengthes, data, 'o-')
%hold on;
%errorbar(seq_lengthes, accuracy_per_tained_rat, std_per_rat)
ylim([0,1])
%xlim ([10,seq_lengthes(end)+])
legend('scrambled', 'inorder')

end    

function [accuracy_mean,accuracy_std] = calculateModelAccuracyOnData(behave_data, model_parameters)
%Given the entire set of training data and a training length, this function
%selects a training sequence and checks the accuracy of the next action.
%It perform several repetitions.

num_samples = length(behave_data);

accurate_iteration = [];
reprtitions = 50;
for i=1:reprtitions
    start_train_ind = randi(num_samples-model_parameters-1,1);
    end_train_ind = start_train_ind + model_parameters;
    accurate_iteration = [accurate_iteration, getModelAccuracy(behave_data, start_train_ind, end_train_ind)];
end

accuracy_mean = mean(accurate_iteration);
accuracy_std = var(accurate_iteration);
end


function correct = getModelAccuracy(behave_data, model_params)

behave_train = behave_data(start_train_ind:end_train_ind-1,:);
behave_test = behave_data(end_train_ind,:);
next_envtype = behave_test(1,5);

next_state_prediction = predictNextState (behave_train, next_envtype);
actual_next_state = behave_test(1,7);

correct = actual_next_state == next_state_prediction;
end

function next_state_prediction = predictNextState (behave_train, next_envtype)
%gets the rat behavior in the train set and the next environment type,
%calculates the model parameters and runs viterbi to estimate the last
%state to return a prediction for the next action.

[estimated_parameters.est_trans_reward, ...
    estimated_parameters.est_trans_noreward, ...
    estimated_parameters.est_emits_homo, ...
    estimated_parameters.est_emits_hetro] = estimateModelParameters( behave_train);

%PredictNextAction()
observation_sequece = behave_train(:,6);
envtype = behave_train(:,5);
reward = behave_train(:,4);

estimatedstates_hmm = mazehmmviterbi(observation_sequece,envtype,reward, ...
    estimated_parameters.est_trans_reward, ...
    estimated_parameters.est_trans_noreward, ...
    estimated_parameters.est_emits_homo, ...
    estimated_parameters.est_emits_hetro);


last_state = estimatedstates_hmm(end);
last_reward = behave_train(end,4);
[next_state_prediction, next_action_prediction] = ...
    predictNextStateandActionByModelParameters(estimated_parameters, last_state, next_envtype, last_reward);


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Helper Functions %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [envtype,seq,states,rewards] = generate_synthetic_sequence(num_trials)

theta = getModelParameters( 0.01 , 'gt2' );
[envtype,seq,states,rewards] = mazehmmgenerate(num_trials, theta.trR, theta.trNR, theta.eH, theta.eT, 0.5, [1 0; 1 0] );

end


function behave_data = convert_synthetic_data_to_behavior_data(envtype,seq,states,rewards)

    %the relevant cue (which is the odor) the selection b corresponds to the 
    chosenRelevantCue = seq;     
    % In configuration 1 the light corresponds to the sequence, i.e.
    % if seq=1 that means the rats selected the light 1 etc.. and the opopsite in configuration 2. 
    chosenIrrelevantCue = (envtype==1).*seq + (envtype==2).*(3-seq);
    
    chosenDoor = zeros(1,length(seq));
  
    behave_data = [chosenRelevantCue; chosenIrrelevantCue; chosenDoor; rewards; envtype; seq; states ]';
end


function [mean, var] = runExperimentOnSequence(seq_lengthes, behave_data)
mean = [];
var = [];
for train_sequence_length = seq_lengthes
    disp (['Processing sequence length: ', num2str(train_sequence_length)])
    [accuracy_mean, accuracy_var] = calculateModelAccuracyOnData(behave_data);
    mean = [mean, accuracy_mean];
    var = [var, accuracy_var];
end
end