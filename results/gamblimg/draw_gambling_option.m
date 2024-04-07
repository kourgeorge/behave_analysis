function [profit, chance] = draw_gambling_option(prob_range)
%DRAW_GAMBLING_OPTION Summary of this function goes here
%   Detailed explanation goes here
% 
%draw a random chance
real_chance = prob_range(1)+rand*(prob_range(2)-prob_range(1));

%claculate the corresponding random balanced profit.
%corresponding_balanced_real_profit = 1/(real_chance + 0.05.*randn());

noised_chance = max([min([prob_range(2),real_chance + 0.1.*randn()]),prob_range(1)]);
corresponding_balanced_real_profit = 1/noised_chance;

%calculate the profit and chance in the log -log space
chance = -round(log10(real_chance),2);
profit = round(log10(corresponding_balanced_real_profit),2);

% profit = rand();
% chance = rand();

end

