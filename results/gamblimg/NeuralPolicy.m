classdef NeuralPolicy < handle
    %POLICY Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        decision_network
    end
    
    methods
        function obj = NeuralPolicy(network)
            %POLICY Construct an instance of this class
            %   Detailed explanation goes here
            obj.decision_network = network;
        end
        
        function action = f(obj, state)            
            action_prob = obj.decision_network(state');
            [~,action] = max(action_prob);
            action = action';
            %action = find(rand<cumsum(action_prob), 1, 'first');
        end
        
        function action_prob = dist(obj, state)
            action_prob = obj.decision_network(state')';
        end
            
    end
end