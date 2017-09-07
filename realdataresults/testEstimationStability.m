function [ ] = testEstimationStability( )
%TESTESTIMATIONSTABILITY Checks the transition matrix estimation stability
%on different guess matrices
%   Detailed explanation goes here

folder = 'C:\Users\gkour\Google Drive\PhD\Behavior Analysis\behavioral data\data\exp1';
rats = {'003','004','019','027','030','031','032'};

[ guess_trans_noreward,  guess_trans_reward, guess_emit_homo, guess_emit_hetro] = getModelParametersGuess( 0.01, 'guess' );

naive_file = fullfile(folder,['030N','.txt']); 
behave_data = loadRatExpData(naive_file);

k=1;
for noiseVal=0:0.2:1
guesses.guess_trans_noreward = NoiseProbabilityMatrix( noiseVal, guess_trans_noreward);
guesses.guess_trans_reward = NoiseProbabilityMatrix( noiseVal, guess_trans_reward);
guesses.guess_emit_homo = NoiseProbabilityMatrix( noiseVal, guess_emit_homo);
guesses.guess_emit_hetro = NoiseProbabilityMatrix( noiseVal, guess_emit_hetro);

[Nest_trans_reward, Nest_trans_noreward, ~, ~] = estimateModelParameters( behave_data, guesses, 0, true);
subplot(2,3,k)
values = {'O1', 'O2','L1', 'L2'};
heatmap(values, values, Nest_trans_reward, 'ColorbarVisible','off')
title(['Noise: ', num2str(noiseVal)])
k=k+1;
end

suptitle(['Transition matrix dependency on the guess (Applying increasing noise on guess). num Trials: ',... 
    num2str(length(behave_data)),' Acc: ', num2str(mean(behave_data(:,4)))])

end

