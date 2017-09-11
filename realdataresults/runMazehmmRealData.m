%States enum:
% O1(1), O2(2), L1(3), L2(4), R(5)

csv_file_path = 'C:\Users\gkour\Google Drive\PhD\Behavior Analysis\behavioral data\odors(1)\006\006_odor1.csv';
behave_data = loadRatExpData(csv_file_path);
theta = estimateModelParameters( behave_data );

[currentState, logP] = mazehmmviterbi(action, envtype, rewards, theta.trR, theta.trNR, theta.eH, theta.eT);

visualizeSelections(envtype, action, rewards,currentState)