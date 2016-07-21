function res = build_exp_data( csv_file_path)
%BUILD_EXP_DATA Summary of this function goes here
%   The fields of the data files: trail	r	w	c	np	cup	ic_1	ic_2
%      1.   trail (t)- the trial number
%      2.   r  -the right door
%      3.   w - the wrong door
%      4.   c - the door chosen by the rodent
%      5.   np - the time (in seconds) elapsed from the stimuli to the selection of the door (the rodent put his nose in the nosepok) - np(t)=time(s1(t))-time(stimuli(t))
%      6.   cup - the time (in seconds) elapsed from the stimuli to the activation of of the sensor s3. cup(t)=time(s3(t))-time(stimuli(t))
%      7.   ic_1 - is the arm which is associated with the irrelevant cue 1 
%      8.   ic_2 - is the arm which is associated with the irrelevant cue 2

% Here we assume that the relevant cue is the odor
data = csvread(csv_file_path,1,0);
res = [];
for i=1:size(data,1)
    
    chosen_door = data(i,4);
    if (isnan(chosen_door) || chosen_door==0)
        continue;   
    end
    
    correct_door = data(i,2);
    is_chosen_door_correct = (chosen_door==correct_door);
    
    if (chosen_door==correct_door)
        chosen_odor_cue = 1; %we want O1 to represent the correct cue.
    else
        chosen_odor_cue = 2;
    end
    
    door_of_light_cue1 = data(i,7); 
    chosen_light_cue = 2 - (chosen_door == door_of_light_cue1);
    
    res = [res; chosen_odor_cue, chosen_light_cue, is_chosen_door_correct];
        
    end
end

