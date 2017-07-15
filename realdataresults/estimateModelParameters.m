function [est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = estimateModelParameters( behave_data, varargin )
%ESTIMATEMODELPARAMETERS 
%Input: a csv contain the behavioral data of the rat. 
%Output: the SCA-HMM model parameters.

if (nargin<2)
    eps = 0.00;
    [ guess_trans_noreward,  guess_trans_reward, guess_emit_homo, guess_emit_hetro] = getModelParametersGuess( eps, true);
else
    guesses = varargin{1};
    guess_trans_noreward = guesses.guess_trans_noreward;
    guess_trans_reward = guesses.guess_trans_reward;
    guess_emit_homo = guesses.guess_emit_homo;
    guess_emit_hetro = guesses.guess_emit_hetro;
end

chosen_relevant_cue = behave_data(:,1);
chosen_irrelevant_cue = behave_data(:,2);

envtype = (chosen_relevant_cue~=chosen_irrelevant_cue) + 1; % if the selected door O1L1 or O2L2 then envtype=1 else envtype=2   
action = chosen_relevant_cue; % The selected action enum equal to the selected odor. 
rewards = behave_data(:,4);

[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = mazehmmtrain(action', envtype', rewards', guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro,'VERBOSE', false, 'maxiterations', 500, 'TOLERANCE',0.01);
 
end

