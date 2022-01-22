function theta_gt = getGTparameters(N)
%GETGTPARAMETERS Summary of this function goes here
%   Detailed explanation goes here
% eps_r = 0.8 + 0.05*randn(1,1);
% eps_nr = 0.2+0.05*randn(1,1);

eps_r = 0.5 + 0.5*rand(1,1);
eps_nr = 0.5*rand(1,1);

theta_gt.trR  = ((1-eps_r)/(N-1))*(ones(N)-eye(N)) + eye(N)*eps_r;
theta_gt.trNR = ((1-eps_nr)/(N-1))*(ones(N)-eye(N)) + eye(N)*eps_nr;

eps = abs(0.1+0.05*randn(1,1));
theta_gt.eH = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

theta_gt.eT = [1-eps eps;
    eps 1-eps;
    eps 1-eps;
    1-eps eps];



% x = 1:-1:0;
% theta_gt.eH = getrandomdistribution(2,2);
% theta_gt.eT = x*theta_gt.eH+(1-x)*inverse(theta_gt.eH);


end

