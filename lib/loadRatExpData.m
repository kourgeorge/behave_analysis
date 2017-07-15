function [res] = loadRatExpData(csv_file_path, odorRelevant)
%LOADRATEXPDATA Summary of this function goes here
%   The fields of the data files: trail	r	w	c	np	cup	ic_1	ic_2
%      1.   trail (t)- the trial number
%      2.   r - the right(correct) door
%      3.   w - the wrong door
%      4.   c - the door chosen by the rodent
%      5.   np - the time (in seconds) elapsed from the stimuli to the selection of the door (the rodent put his nose in the nosepok) - np(t)=time(s1(t))-time(stimuli(t))
%      6.   cup - the time (in seconds) elapsed from the stimuli to the activation of of the sensor s3. cup(t)=time(s3(t))-time(stimuli(t))
%      7.   ic_1 - is the arm which is associated with the irrelevant cue 1 
%      8.   ic_2 - is the arm which is associated with the irrelevant cue 2

%   The fields in the output are as follows:
%      1. The choosen relevant cue (for instance odor, 1 mean that the choosen door had odor 1)
%      2. The choosen irrelevant cue (for instance color, 2 mean that the choosen door had color 2)
%      3. The number of the door that was chosen
%      4. Is the choosen door is the correct door (rewards).
%      5. The environment type 1=(O1L1,O2,L2) and 2=(O1L2,O2,L1)
%      6. The choosen option

data = csvread(csv_file_path,1,0);
data(data(:,4)==0,:) = [];
numTrials = size(data,1);
res = zeros(numTrials, 6);
behave_struct = struct;
for i=1:numTrials
    
    chosen_door = data(i,4);
    correct_door = data(i,2);
    door_of_irrelevant_cue1 = data(i,7);
    door_of_relevant_cue1 = correct_door; % becase we assume that cue1 is the correct door.
    
    if (isnan(chosen_door) || chosen_door==0)
        continue;   
    end
    
    
    reward = (chosen_door==correct_door);
    
    if (chosen_door==correct_door)
        chosen_relevant_cue = 1;
    else
        chosen_relevant_cue = 2;
    end
    
     
    chosen_irrelevant_cue = 2 - (chosen_door == door_of_irrelevant_cue1);
    
    if (door_of_relevant_cue1==door_of_irrelevant_cue1)
        envtype=1;
    else
        envtype=2;
    end
    
    chosen_option = chosen_relevant_cue;  % because the selected odor number corresponds to the selected option number
    res(i,:) = [chosen_relevant_cue, chosen_irrelevant_cue, chosen_door, reward, envtype, chosen_option];
    
end

% behave_struct.chosenRelevantCue = res(:,1);
% behave_struct.chosenIrrelevantCue = res(:,2);
% behave_struct.chosenDoor= res(:,3);
% behave_struct.reward = res(:,4);
% behave_struct.envtype = res(:,5);
% behave_struct.chosenOption = res(:,1); % because the selected odor number corresponds to the selected option number
end

