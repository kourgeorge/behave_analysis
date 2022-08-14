classdef GamblingAgent < handle
    properties
        TrR {mustBeNumeric}
        TrnR {mustBeNumeric}
        Policies
        curr_state {mustBeNumeric}
    end
    methods
        function obj = GamblingAgent(trR, trnr, policies)
            
            obj.TrR = trR;
            obj.TrnR = trnr;
            obj.curr_state = 1;
            obj.Policies = policies;
            
        end
        
        function action = act(obj, u)
            
            action = obj.Policies{obj.curr_state}.f(u);
            %[obj.curr_state,action]
            %action = GamblingAgent.draw(act_dist) ;
        end
        
        
        function feedback(obj,reward)
            if reward ==1
                obj.curr_state = GamblingAgent.draw(obj.TrR(obj.curr_state,:));
            else
                obj.curr_state = GamblingAgent.draw(obj.TrnR(obj.curr_state,:));
            end
            
        end
    end
    
    methods(Static)
    
        function item = draw(p)
            item = find(rand<cumsum(p), 1, 'first');
        end
        
    end
end
