classdef AntiLinearPolicy < handle
    %POLICY Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        a
        b
    end
    
    methods
        function obj = AntiLinearPolicy(a,b)
            %POLICY Construct an instance of this class
            %   Detailed explanation goes here
            obj.a = a;
            obj.b = b;
        end
        
        function action = f(obj, state)
            action_prob = dist(obj, state);
            [~,action] = max(action_prob, [], 2) ;
        end
        
         function action_prob = dist(obj, state)
%             log_prob = log10(state(2));
%             log_profit = log10(state(1));
            x = state(:,1);
            y = state(:,2);
            %action = (chance > obj.a*profit+obj.b)+1;
            %action_prob = onehot(action,1:2);
            border_y = obj.a*x+obj.b;
            action_prob = softmax([zeros(size(state,1),1), border_y-y]')';
            
             
%             log_prob = log10(state(2));
%             log_reward = log10(state(1));
%             
%             if (log_prob+log_reward)>log10(obj.b)
%                 action_prob=[0,1];
%             else
%                 action_prob=[1,0];
%             end
        end
    end
end

