function [ guess_trans_noreward,  guess_trans_reward, guess_emit_homo, guess_emit_hetro] = getModelParametersGuess( eps , uniform )
%GETMODELPARAMETERSGUESS Summary of this function goes here
%   Detailed explanation goes here

guess_trans_noreward = [0.5 0.3 0.1 0.1
           0.3 0.5 0.1 0.1
           0.1 0.1 0.5 0.3
           0.1 0.1 0.3 0.5];
guess_trans_reward = [0.85 0.05 0.05 0.05
           0.05 0.85 0.05 0.05 
           0.05 0.05 0.85 0.05 
           0.05 0.05 0.05 0.85];

if (uniform)
    guess_trans_noreward = 0.25*ones(4);
    guess_trans_reward = 0.25*ones(4);
end

guess_emit_homo = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps];
         
guess_emit_hetro = [1-eps eps;
             eps 1-eps;
             eps 1-eps;
             1-eps eps];
         
end

