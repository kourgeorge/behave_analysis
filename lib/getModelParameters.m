function theta = getModelParameters(eps, type)
%GETMODELPARAMETERS This function returns a intial model parameters.
% The emission parameters are less interesting in our setup since the underlying
% strategy and the current environemtn configuration decides
% determisitically the output.
%   eps - the eps for the emission matrices.
%   type - 'guess' initial sensible guess
%          'random' - a random matrix,
%          'uniform' - a uniform trannsion matrix


if strcmp(type,'guess') || strcmp(type,'gt')
    theta.trR = [0.85 0.05 0.05 0.05
        0.05 0.85 0.05 0.05
        0.05 0.05 0.85 0.05
        0.05 0.05 0.05 0.85];
    theta.trNR = [0.4 0.3 0.15 0.15
        0.3 0.4 0.15 0.15
        0.15 0.15 0.4 0.3
        0.15 0.15 0.3 0.4];

end


if strcmp(type,'guess2') || strcmp(type,'gt2')
    theta.trR = [0.7 0.1 0.1 0.1
        0.1 0.7 0.1 0.1
        0.1 0.1 0.7 0.1
        0.1 0.1 0.1 0.7];
    theta.trNR = [0.1 0.3 0.3 0.3
        0.3 0.1 0.3 0.3
        0.3 0.3 0.1 0.3
        0.3 0.3 0.3 0.1];

end


if strcmp(type,'uniform')
    theta.trR = 0.25*ones(4);
    theta.trNR = 0.25*ones(4);
end

if strcmp(type, 'random')
    theta.trR = getrandomdistribution(4,4);
    theta.trNR = getrandomdistribution(4,4);
end

theta.eH = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

theta.eT = [1-eps eps;
    eps 1-eps;
    eps 1-eps;
    1-eps eps];

end

