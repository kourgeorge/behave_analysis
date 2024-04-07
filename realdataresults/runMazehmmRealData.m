%States enum:
% O1(1), O2(2), L1(3), L2(4)

csv_file_path = 'C:\Users\gkour\Google Drive\PhD\IBAGS paper\BEHAVIOR_rat004\ODORS\004_26012016.csv';

behave_data = loadRatExpData(csv_file_path);
[theta,action,rewards,envtype] = estimateModelParameters( behave_data );

[currentState, logP] = mazehmmviterbi(action, envtype, rewards, theta.trR, theta.trNR, theta.eH, theta.eT);

visualizeSelections(envtype, action, rewards,currentState)