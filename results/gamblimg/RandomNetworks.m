function policies = RandomNetworks(N)


policies = [];
states_sample = [0.1, 10; 0.25, 4];
actions_sample = [0, 1; 1, 0];

for i=1:N
    %net1 = lvqnet();
    net1 = patternnet(5);
    %net1.trainParam.epochs = 10;
    net1.trainParam.showWindow = false;
    policies = [policies; {configure(net1,states_sample,actions_sample)}];
end
end