function [chance,profit] = draw_gambling_option(prob_range)
%DRAW_GAMBLING_OPTION Summary of this function goes here
%   Detailed explanation goes here


% minus_log_prob = prob_range(1)+rand*(prob_range(2)-prob_range(1));
% 
% corresponding_balanced_profit = minus_log_prob + 0.05.*randn();
% %profit = round(10^corresponding_balanced_profit, 1);
% 
% profit = round(10^(rand()), 2);
% chance = round(10^(-minus_log_prob), 2);

profit = rand();
chance=rand();

end

