function theta_real = getConstantGTparameters()
%GETCONSTANTGTPARAMETERS Summary of this function goes here
%   Detailed explanation goes here

eps = 0.05;
theta_real.eH = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

theta_real.eT = [1-eps eps; %o1l2 o2l1
    eps 1-eps;
    eps 1-eps;
    1-eps eps];


% theta_real.trR = ...
%     [0.8 0.1 0.05 0.05;
%     0.1 0.8 0.05 0.05;
%     0.05 0.05 0.8 0.1;
%     0.05 0.05 0.1 0.8];
% 
% 
% theta_real.trNR = ...
%     [0.1 0.5 0.2 0.2;
%     0.5 0.1 0.2 0.2;
%     0.2 0.2 0.1 0.5;
%     0.2 0.2 0.5 0.1];


theta_real.trR = ...
    [0.9 0.1/3 0.1/3 0.1/3;
    0.1/3 0.9 0.1/3 0.1/3;
   0.1/3 0.1/3 0.9 0.1/3;
    0.1/3 0.1/3 0.1/3 0.1/3];


theta_real.trNR = ...
    [0.1 0.4 0.15 0.15;
    0.4 0.1 0.15 0.15;
    0.15 0.15 0.1 0.4;
    0.15 0.15 0.4 0.1];

end
