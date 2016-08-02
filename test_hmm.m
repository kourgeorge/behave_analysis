%At the beginning do not use the reward.

%States enum:
% O1(1), O2(2), L1(3), L2(4), R(5)

%Emits enum:
%O1L1(1), O1L2(2), O2L1(3), O2L2(4)


%create an observations matrix, assume 3 paths of length 5
%O1L1, O2L1, O2L2, O2L1, O2L1 (L1, L1, O2, O2, O2)
%O2L1, O2L2, O1L2, O1L2, O2L2 (O2, O2, L2, L2, L2)
%O1L2, O1L1, O2L1, O1L1, O1L2 (O1, L1, L1, L1, O1)

%seqs = [1 3 4 3 1;
%       3 2 2 2 4;
%       2 3 3 1 2];

%create a transition matrix (5 states - 5X5 matrix)
guess_trans_noreward = [0.6 0.25 0.05 0.05 0.05;
           0.25 0.6 0.05 0.05 0.05;
           0.05 0.05 0.6 0.25 0.05;
           0.05 0.05 0.25 0.6 0.05
           0.2 0.2 0.2 0.2 0.2];
guess_trans_reward = [0.8 0.05 0.05 0.05 0.05;
           0.05 0.8 0.05 0.05 0.05;
           0.05 0.05 0.8 0.05 0.05;
           0.05 0.05 0.05 0.8 0.05
           0.2 0.2 0.2 0.2 0.2];

%O1L1	O1L2	O2L1	O2L2
%O1     0.5		0.5		0		0
%O2     0		0		0.5		0.5
%L1     0.5		0		0.5		0		
%L2     0		0.5		0		0.5
%R      0.25	0.25	0.25	0.25

%create an emiision matrix (4 emissions, 5 states - 5X4 matrix)
eps = 0.05;
guess_emit_homo = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps;
             0.5 0.5];
         
guess_emit_hetro = [1-eps eps;
             eps 1-eps;
             eps 1-eps;
             1-eps eps;
             0.5 0.5];
         

         
% on real data
csv_file_path = 'C:\Users\kour\OneDrive - University of Haifa\Experimental data\BEHAVIOR\ODORS_2_odorsrelevantsecoindpair_palmarosa for reward_bohno connect with reward\RAT2_0102.csv';
         
selected_cues_reward = build_exp_data(csv_file_path);
num_trials = size (selected_cues_reward,1);
observation_reward = zeros(num_trials,3);
for i=1: num_trials
    exptype = (selected_cues_reward(i,1)~=selected_cues_reward(i,2)) + 1;
    action = selected_cues_reward(i,1);
    reward = selected_cues_reward(i,3);
    observation_reward(i,:)=[action, exptype, reward];
end
emission_seq = observation_reward(:,1)';
envtype = observation_reward(:,2)';
rewards = observation_reward(:,3)';

[est_trans_reward, est_trans_noreward , est_emits_homo, est_emits_hetro] = myhmmtrain(emission_seq, envtype , rewards ,guess_trans_reward ,guess_trans_noreward ,guess_emit_homo, guess_emit_hetro,'VERBOSE',true, 'maxiterations', 1500);




