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
guess_trans_noreward = [0.5 0.3 0.1 0.1
           0.3 0.5 0.1 0.1
           0.1 0.1 0.5 0.3
           0.1 0.1 0.3 0.5];
guess_trans_reward = [0.85 0.05 0.05 0.05
           0.05 0.85 0.05 0.05 
           0.05 0.05 0.85 0.05 
           0.05 0.05 0.05 0.85];

%O1L1	O1L2	O2L1	O2L2
%O1     0.5		0.5		0		0
%O2     0		0		0.5		0.5
%L1     0.5		0		0.5		0		
%L2     0		0.5		0		0.5
%R      0.25	0.25	0.25	0.25

%create an emiision matrix (4 emissions, 5 states - 5X4 matrix)
eps = 0.001;
guess_emit_homo = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps];
         
guess_emit_hetro = [1-eps eps;
             eps 1-eps;
             eps 1-eps;
             1-eps eps];
         

% guess_trans_noreward = rand(4,4);
% guess_trans_reward = rand(4,4);
% guess_emit_homo = rand(4,2);
% guess_emit_hetro = rand(4,2);

% on real data
%csv_file_path = 'C:\Users\gkour\Google Drive\PhD\Experimental data\BEHAVIOR\ODORS_2_odorsrelevantsecoindpair_palmarosa for reward_bohno connect with reward\RAT2_0102.csv';
csv_file_path = 'C:\Users\gkour\Google Drive\PhD\Behavior Analysis\behavioral data_\odors(1)\007\007_odor1.csv';
%csv_file_path = 'C:\Users\gkour\Google Drive\PhD\Behavior Analysis\behavioral data_\odors(1)\008\008_odor1.csv';
selected_cues_reward = buildRatExpData(csv_file_path);
num_trials = size (selected_cues_reward,1);

envtype = (selected_cues_reward(:,1)~=selected_cues_reward(:,2)) + 1; % if the selected door O1L1 or O2L2 then envtype=1 else envtype=2   
action = selected_cues_reward(:,1); % The selected action enum equal to the selected odor. 
rewards = selected_cues_reward(:,4);

[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = mazehmmtrain(action, envtype, rewards ,guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro,'VERBOSE', true, 'maxiterations', 1500);

%test the viterbi if it makes sense with reasonable transition matrices.
[currentState, logP] = mazehmmviterbi(action, envtype, rewards, guess_trans_reward, guess_trans_noreward,est_emits_homo, est_emits_hetro);


[currentState, logP] = mazehmmviterbi(action, envtype, rewards, est_trans_reward, est_trans_noreward,est_emits_homo, est_emits_hetro);