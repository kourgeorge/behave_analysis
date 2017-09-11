function visualizeSelections( envtypes, actions, rewards,estimsted_states )
%VISUALIZESELECTIONS Summary of this function goes here
%   Detailed explanation goes here


selections = [];
for i=1:length(actions)
    envtype = envtypes(i);
    action = actions(i);
    if envtype==1
        if action==1
            selections = [selections; 1,3];
        else
            selections = [selections; 2,4];
        end
    else
        if action==1
            selections = [selections; 1,4];
        else
            selections = [selections; 2,3];
        end 
    end
end


start_window = 1;
end_window = length(actions);
selections = selections(start_window:end_window,:);

scatter (start_window:end_window, selections(:,1))
hold on;
scatter (start_window:end_window, selections(:,2))

scatter (start_window:end_window, estimsted_states(start_window:end_window), 'filled', 'g')
strategy = ['O1'; 'O2'; 'L1'; 'L2'];
set(gca,'YLim',[0 4],'YTick',1:4,'YTickLabel',strategy);

mean(rewards)
end

