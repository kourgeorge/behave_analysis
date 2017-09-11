function theta = estimateModelParameters( behave_data, varargin )
%ESTIMATEMODELPARAMETERS 
%Input: a csv contain the behavioral data of the rat. 
%Output: the SCA-HMM model parameters.


max_iterations = 500;
tolerance = 1e-4; 

if (nargin<2)
    eps = 0.00;
    theta_guess = getModelParameters( eps, 'uniform');
else
    guesses = varargin{1};
    theta_guess =  guesses;
end

chosen_relevant_cue = behave_data(:,1);
chosen_irrelevant_cue = behave_data(:,2);

envtype = (chosen_relevant_cue~=chosen_irrelevant_cue) + 1; % if the selected door O1L1 or O2L2 then envtype=1 else envtype=2   
action = chosen_relevant_cue; % The selected action enum equal to the selected odor. 
rewards = behave_data(:,4);

[theta.trR, theta.trNR, theta.eH, theta.eT] = mazehmmtrain(action', envtype', rewards', theta_guess.trR, theta_guess.trNR, theta_guess.eH, theta_guess.eT,'VERBOSE', false, 'maxiterations', max_iterations, 'TOLERANCE', tolerance);
 
end

