%test myhmmdecode by comparing it to hmmdecode. The comparison is based n
%the fact that both should return the same values, if only ine environmnt
%is used and always there is a reward.

eps = 0.05;
guessEhomo = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps;
             0.5 0.5];
guessEhetro = [1-eps eps;
             eps 1-eps;
             eps 1-eps;
             1-eps eps;
             0.5 0.5];
         
guessTRr = [0.8 0.05 0.05 0.05 0.05;
           0.05 0.8 0.05 0.05 0.05;
           0.05 0.05 0.8 0.05 0.05;
           0.05 0.05 0.05 0.8 0.05
           0.2 0.2 0.2 0.2 0.2];
guessTRnr = [0.6 0.25 0.05 0.05 0.05;
           0.25 0.6 0.05 0.05 0.05;
           0.05 0.05 0.6 0.25 0.05;
           0.05 0.05 0.25 0.6 0.05
           0.2 0.2 0.2 0.2 0.2];

       
num_trails = 10;
exptype = ones(1, num_trails); 
reward = ones(1, num_trails);
[seq,states] = hmmgenerate(num_trails,guessTRr,guessEhomo);


%% test 1 - compare between myhmmdecode and the original hmmdecode.
% bth should give the same result on the special case when thhere is only
% one envtype and always the agents recieves a reward.
[pstates,logPseq,fs,bs,scale] = myhmmdecode(seq,exptype,reward,guessTRr, guessTRnr,guessEhomo,guessEhetro);
[opstates,ologPseq,ofs,obs,oscale] = hmmdecode(seq,guessTRr,guessEhomo);