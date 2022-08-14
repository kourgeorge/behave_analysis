classdef GamblingGame < handle
    %GAMBLINGGAME Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        prob_range = [0.1, 1]; 
        total_rounds = 40
        states = []
        actions = []
        rewards = []
        wins = []
        hidden_states = []
    end
    
    methods
        function  obj = GamblingGame (L)
             obj.total_rounds = L; 
        end
        
        function start(obj, agent)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            for i=1:obj.total_rounds
                game = obj.drawoptions();
                state = game(2,:);
                obj.states = [obj.states; state];
                obj.hidden_states = [obj.hidden_states, agent.curr_state];
                action = agent.act(state);
                obj.actions = [obj.actions, action];
                win_probability = game(action,2);
                    if rand < win_probability
                        obj.rewards = [obj.rewards,game(action,1)];
                        obj.wins = [obj.wins,1];
                    else
                        obj.rewards = [obj.rewards,0];
                        obj.wins = [obj.wins,0];
                    end 
                agent.feedback(obj.wins(end))
            end
        end
        
        
        function optins = drawoptions(obj)
            [chance,value]  = draw_gambling_option(obj.prob_range);
            optins = [1,1; value, chance];
            %game(:,2) = -game(:,2);
            
        end
        
        function draw_empirical_decision_boundary(obj)
            figure;
            allstates = unique(obj.hidden_states);
            for state=allstates
                subplot(length(allstates),1,state)
                state_trials = find(obj.hidden_states==state);
                scatter(obj.states(state_trials,1), obj.states(state_trials,2), [] , 15.*obj.actions(state_trials)+20,'filled')
                hold on;
            end
            hold off;

        end
        
    end
end
    
