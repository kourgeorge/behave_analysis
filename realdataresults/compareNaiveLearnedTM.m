function compareNaiveLearnedTM()
%COMPARENAIVELEARNEDTM Calculate the transition matrices of both trained
%and naive rats for both rewarded and unrewarded trials. 
%The transition matrices are calculated based on the SCA-HMM model. 
%The average transition probability is shown for the several rats.  

folder = 'C:\Users\gkour\Google Drive\PhD\Behavior Analysis\behavioral data\data\exp1';
rats = {'003','004','019','027','030','031','032'};

TNest_trans_reward = zeros(4,4);
TLest_trans_reward = zeros(4,4);
TNest_trans_noreward = zeros(4,4);
TLest_trans_noreward = zeros(4,4);

for rat=rats
   naive_file = fullfile(folder,[rat{1},'N','.txt']); 
   learned_file = fullfile(folder,[rat{1},'.txt']);
   behave_data = loadRatExpData(naive_file);
   theta_naive = estimateModelParameters( behave_data );
   behave_data = loadRatExpData(learned_file);
   theta_trained= estimateModelParameters( behave_data );
   TNest_trans_reward = TNest_trans_reward + theta_naive.trR;
   TLest_trans_reward = TLest_trans_reward + theta_trained.trR;
   TNest_trans_noreward = TNest_trans_noreward + theta_naive.trNR;
   TLest_trans_noreward = TLest_trans_noreward +theta_trained.trNR;
   
end

values = {'O1', 'O2','L1', 'L2'};
subplot(2,2,1)
heatmap(values, values, TNest_trans_reward/7, 'ColorbarVisible','off')
title ('Naive rats in rewarded trials')
subplot(2,2,2)
heatmap(values, values, TLest_trans_reward/7, 'ColorbarVisible','off')
title ('Trained rats in rewarded trials')
subplot(2,2,3)
heatmap(values, values, TNest_trans_noreward/7, 'ColorbarVisible','off')
title ('Naive rats in non rewarded trials')
subplot(2,2,4)
heatmap(values, values, TLest_trans_noreward/7, 'ColorbarVisible','off')
title ('Trained rats in non rewarded trials')

suptitle('Comparing transition matrices of naive and trained rats, (Y->t-1) and (X->t)')
end

