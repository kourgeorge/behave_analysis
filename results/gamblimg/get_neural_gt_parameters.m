function theta_gt= get_neural_gt_parameters()

%This works ok
theta_gt.trR  = [0.9 0.1; 0.7, 0.3];
theta_gt.trNR = [0.3 0.7; 0.1, 0.9];


a1 = wblrnd(1,1.5);
c1(1) = 0.5*rand(1);
c1(2) = 0.5+0.5*rand(1);
b1 = c1(2)-a1*c1(1);


a2 = 0.5*a1;
c2(1) = 0.5+0.5*rand(1);
c2(2) = 0.5*rand(1);
b2 = c2(2)-a2*c2(1);


% select a new c beween x inter.
% min_x = max([0,-b1/a1]);
% c2(1) = min_x + (1-min_x)*rand();
% 
% max_y = a1*c2(1)+b1;
% c2(2) = (max_y)*rand();
% 
% a2 = 0.5*a1;
% b2 = c2(2)-a2*c2(1);

%b2 = b1-a1*0.3;
% a2 = wblrnd(2,1.5);
% c2 = rand(1,2);
% b2 = c2(2)-a2*c2(1);
% 
%d = 1
% 

% x=0:0.01:1;
% figure;
% hold on
% plot(x,a1*x+b1);
% ylim([0,1])
%plot(x,a2*x+b2);
%ylim([0,1])

%theta_gt.policies = {LinearPolicy(1,0.5);LinearPolicy(0,0.5)};
theta_gt.policies = {LinearPolicy(a1,b1);LinearPolicy(a2,b2)};
end

