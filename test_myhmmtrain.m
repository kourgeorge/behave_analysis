%test myhmmm by giving it a single experiment type and a always reward and
%compare the result with the original hmm

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

%% test 1 - create sequence of a single environemnt type (h) 
% and reward and make sure the reduction of the
% generalized implementation (myhmmtrain) emits the same results as the original
% hmm. 
% generate the sequnce using hmmgenerate (no myhmmgenrate).
num_trails = 500;
envtype = ones(1, num_trails); 
rewards = ones(1, num_trails);
[emission_seq,states] = hmmgenerate(num_trails,guess_trans_noreward,guess_emit_hetro);
[est_trans_reward, est_trans_noreward , est_emits_homo, est_emits_hetro] = myhmmtrain(emission_seq, envtype , rewards ,guess_trans_reward ,guess_trans_noreward ,guess_emit_homo, guess_emit_hetro,'VERBOSE',false, 'maxiterations', 1500);
[est_trans, est_emits] = hmmtrain(emission_seq , guess_trans_reward, guess_emit_homo,'VERBOSE',false, 'maxiterations', 1500);

if (est_trans_reward ==  est_trans)
    fprintf('Pass\n')
else
    fprintf('Fail\n')
end

