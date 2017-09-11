function [ est_trans_reward_avg_naive, est_trans_reward_avg_learned ] = compareTransitionMatrices( folder_path )
%COMPARETRANSITIONMATRICES Compare the estimated transition matrices before
%and after learning
%   Input: the folder containing the csv files for each rat in each day
%   Output: the average transition matrices before and after learning.

est_trans_reward_avg_learned = [];
est_trans_reward_avg_naive = [];
num_learned = 0;
num_naive = 0;
files = dir([folder_path,'/*.csv']);

for file = files'
    file_path = fullfile(file.folder,file.name);
    behave_data = loadRatExpData(file_path);
    theta = estimateModelParameters( behave_data );

    if (learning_score(behave_data)> 0.7)
        est_trans_reward_avg_learned = est_trans_reward_avg_learned + theta.trR;
        num_learned = num_learned+1;
    else
        est_trans_reward_avg_naive = est_trans_reward_avg_naive + theta.trR;
        num_naive = num_naive+1;
    end
    
est_trans_reward_avg_learned = est_trans_reward_avg_learned/num_learned;
est_trans_reward_avg_naive = est_trans_reward_avg_naive/num_naive;

end

function score = learning_score(behave_data)
rewards = behave_data(:,4);
score = mean(rewards);