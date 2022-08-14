function interaction = generate_gambling_trajectory(trR, trNR, policies, L)
%GENERATE_GAMBLING_TRAJECTORY Summary of this function goes here
%   Detailed explanation goes here

agent = GamblingAgent(trR, trNR, policies);
interaction = GamblingGame(L);
interaction.start(agent)

end

