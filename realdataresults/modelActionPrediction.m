function [] = modelActionPrediction()
%MODELACTIONPREDICTION Calculates the next step prediction accuracy based
%on a training set that is used to calculate the model parameters. The
%goal is to find the optimal length of the training set, that is not to
%short to allow accurate training of the model and is not too long to
%include in the model several traisitions matrices that changes due to the 
% training of the rat).

%Given a rat with N trails, perform the following for K=20:5:80
%Choose a sub-sequence of length K, and train the model.
%Based on the trained model estimate the next action of the rat given also the state in the last trial of the training sequence (calculated using the viterbi algorithm), the current environment type, and whether a reward was given in the last trial.
%Perform 100 repetitions.

folder = 'C:\Users\gkour\Google Drive\PhD\Behavior Analysis\behavioral data\data\exp1';
rats = {'004','019','027','030', '031','032'};

sequence_length_acc_trained = [];
sequence_length_acc_naive = [];
seq_lengthes = 15:5:35;

%for each rat
for i=1:length(rats)
    rat = rats{i};

    trained_file = fullfile(folder,[rat,'.txt']); 
    trained_behave_data = loadRatExpData(trained_file);
    disp (['Processing Trained rat ',rat,' with sequence length: ', num2str(length(trained_behave_data))])
    
    [accuracy_per_tained_rat, var_per_rat] = runExperimentOnRat(seq_lengthes, trained_behave_data);
    sequence_length_acc_trained = [sequence_length_acc_trained; accuracy_per_tained_rat];
    
    naive_file = fullfile(folder,[rat,'N.txt']); 
    naive_behave_data = loadRatExpData(naive_file);
    disp (['Processing Naive rat ',rat,' with sequence length: ', num2str(length(naive_behave_data))])
    
    [accuracy_per_naive_rat, var_per_rat] = runExperimentOnRat(seq_lengthes, naive_behave_data);
    sequence_length_acc_naive = [sequence_length_acc_naive; accuracy_per_naive_rat];
    
    plotresult(i, seq_lengthes, [accuracy_per_naive_rat', accuracy_per_tained_rat']);
    title(['Rat ',rats{i},'. #Naive: ', num2str(length(naive_behave_data)), '. #Trained: ', num2str(length(trained_behave_data))])
    
end

plotresult(8, seq_lengthes, [nanmean(sequence_length_acc_naive)', nanmean(sequence_length_acc_trained)']);
title('Average Accuracy')

suptitle('Next step prediction accuracy vs. training sequence length.')

end

function plotresult(sub, seq_lengthes, data)

subplot(2,4,sub)
plot(seq_lengthes, data, 'o-')
%hold on;
%errorbar(seq_lengthes, accuracy_per_tained_rat, std_per_rat)
ylim([0,1])
xlim ([10,40])
legend('naive', 'trained')

end    

function [mean, var] = runExperimentOnRat(seq_lengthes, behave_data)
mean = [];
var = [];
for train_sequence_length = seq_lengthes
    disp (['Processing sequence length: ', num2str(train_sequence_length)])
    if (length(behave_data)-1 <= train_sequence_length)
        disp(['Rat has ', num2str(length(behave_data)), ' trials. Needed: ', num2str(train_sequence_length)])
        mean = [mean, NaN];
        var = [var, NaN];
        continue;
    end
    [accuracy_mean, accuracy_var] = calculateModelAccuracyOnData(behave_data, train_sequence_length);
    mean = [mean, accuracy_mean];
    var = [var, accuracy_var];
end

nd

function [accuracy_mean,accuracy_std] = calculateModelAccuracyOnData(behave_data, train_sequence_length)
%Given the entire set of training data and a training length, this function
%selects a training sequence and checks the accuracy of the next action.
%It perform several repetitions.

num_samples = length(behave_data);

accurate_iteration = [];
reprtitions = 50;
for i=1:reprtitions
    start_train_ind = randi(num_samples-train_sequence_length-1,1);
    end_train_ind = start_train_ind + train_sequence_length;    
    reward_on_next_step = behave_data(end_train_ind+1,4);
    %check if the rat made an error in trial t+K+1
    %while (reward_on_next_step==true)
    %    start_train_ind = randi(num_samples-train_sequence_length-1,1);
    %    end_train_ind = start_train_ind + train_sequence_length;
    %    reward_on_next_step = behave_data(end_train_ind+1,4);
    %end
    accurate_iteration = [accurate_iteration, getModelAccuracy(behave_data, start_train_ind, end_train_ind)];
end

accuracy_mean = mean(accurate_iteration);
accuracy_std = var(accurate_iteration);
end

function correct = getModelAccuracy(behave_data, start_train_ind, end_train_ind)

behave_train = behave_data(start_train_ind:end_train_ind,:);
behave_test = behave_data(end_train_ind+1,:);
next_envtype = behave_test(1,5);

next_action_prediction = predictNextAction (behave_train, next_envtype);
actual_next_action = behave_test(1,6);

correct = actual_next_action == next_action_prediction;
end

function next_action_prediction = predictNextAction (behave_train, next_envtype)
%gets the rat behavior in the train set and the next environment type,
%calculates the model parameters and runs viterbi to estimate the last
%state to return a prediction for the next action.

theta = estimateModelParameters( behave_train );
 

%PredictNextAction()
observation_sequece = behave_train(:,6);
envtype = behave_train(:,5);
reward = behave_train(:,4);

estimatedstates_hmm = mazehmmviterbi(observation_sequece,envtype,reward,...
    theta.trR, theta.trNR, theta.eH, theta.eT);

last_state = estimatedstates_hmm(end);

last_reward = behave_train(end,4);
next_action_prediction = predictNextActionByModelParameters(theta, last_state, next_envtype, last_reward);


end

function next_action = predictNextActionByModelParameters(theta, last_state, next_envtype, last_reward)
%gets the model parameters environment type and last reward and returns the
%next action.

if (last_reward)
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