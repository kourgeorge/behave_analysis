function [ trans_noreward,  trans_reward, emit_homo, emit_hetro] = getModelParameters( eps , type )
%GETMODELPARAMETERS This function returns a intial model parameters.
% The emission parameters are less interesting in our setup since the underlying
% strategy and the current environemtn configuration decides
% determisitically the output.
%   eps - the eps for the emission matrices.
%   type - 'guess' initial sensible guess
%          'random' - a random matrix,
%          'uniform' - a uniform trannsion matrix

if strcmp(type,'guess')
    trans_noreward = [0.4 0.3 0.15 0.15
        0.3 0.4 0.15 0.15
        0.15 0.15 0.4 0.3
        0.15 0.15 0.3 0.4];
    trans_reward = [0.85 0.05 0.05 0.05
        0.05 0.85 0.05 0.05
        0.05 0.05 0.85 0.05
        0.05 0.05 0.05 0.85];
end


if strcmp(type,'uniform')
    trans_noreward = 0.25*ones(4);
    trans_reward = 0.25*ones(4);
end

if strcmp(type, 'random')
    trans_noreward = getrandomdistribution(4,4);
    trans_reward = getrandomdistribution(4,4);
end

emit_homo = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

emit_hetro = [1-eps eps;
    eps 1-eps;
    eps 1-eps;
    1-eps eps];

end

