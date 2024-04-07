function distance = CompareGamblingPolicies(policy1 ,policy2, iterations)
%COMPARE_PREFERENCE_CURVE Summary of this function goes here
%   Detailed explanation goes here

states = zeros(2,iterations);
for i=1:iterations
    [chance,profit]=draw_gambling_option([0.1,1]);
    states(:,i)=[chance;profit];
end
action_dist1 = policy1.dist(states');
action_dist2 = policy2.dist(states');
div_lst= JSDiv(action_dist1,action_dist2);


actions1 = policy1.f(states');
actions2 = policy2.f(states');
%div_lst= (actions1==actions2)*1;



% figure;
% subplot(2,1,1);
% [~,actions] = max(action_dist1');
% scatter(states(1,:), states(2,:), [] , 15.*actions+20,'filled')
% 
% subplot(2,1,2);
% [~,actions] = max(action_dist2');
% scatter(states(1,:), states(2,:), [] , 15.*actions+20,'filled')
% hold on;

%      figure;
%     a1=policy1.f(log10([chance, profit]));
%     a2=policy2.f(log10([chance, profit]));
%     scatter(profit, chance, [], (a1==a2)*10+3);
%     hold on
%


% x=1:0.01:10;
% y = 1./x;
% plot (x,y)
%
%hold off;
distance = mean(div_lst);
end

