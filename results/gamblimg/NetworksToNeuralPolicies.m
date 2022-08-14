function neuralpolices = NetworksToNeuralPolicies(networks)
%NETWORKSTONEURALPOLICIES Summary of this function goes here
%   Detailed explanation goes here
neuralpolices = cell(length(networks),1);
for i=1:length(networks)
    neuralpolices{i} = NeuralPolicy(networks{i});
end
end

