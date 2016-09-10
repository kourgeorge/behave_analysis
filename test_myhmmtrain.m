function test_myhmmtrain()

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
       
test1(guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro);
test2(guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro);

end

function test1(guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro)
% TEST 1 - create sequence of a single environemnt type (h) 
% and reward and make sure the reduction of the
% generalized implementation (myhmmtrain) emits the same results as the original
% hmm. 
% generate the sequnce using hmmgenerate (no myhmmgenrate).
num_trails = 500;
envtype = ones(1, num_trails); 
rewards = ones(1, num_trails);
[emission_seq,states] = hmmgenerate(num_trails,guess_trans_noreward,guess_emit_hetro);
[est_trans_reward, est_trans_noreward , est_emits_homo, est_emits_hetro] = ...
    myhmmtrain(emission_seq, envtype , rewards ,guess_trans_reward ,guess_trans_noreward ...
    ,guess_emit_homo, guess_emit_hetro,'VERBOSE',false, 'maxiterations', 1500);
[est_trans, est_emits] = hmmtrain(emission_seq , guess_trans_reward, guess_emit_homo,'VERBOSE',false, 'maxiterations', 1500);

if (est_trans_reward ==  est_trans)
    fprintf('Test 1 - Pass\n')
else
    fprintf('Test 1 - Fail\n')
end

end

function test2(guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro)
% TEST 2 - generate data using myhmmgeneratedata. The data is only homo 
% environment and no reward. Compare hmmtrain and myhmmtrain. The
% difference in the transition probabilitis and the emission probability
% estimation be close if not identical.

tol = 0.001;
num_trails = 500;
% on synthetic data with no rewarded states and only homo env type.
[envtype,emission_seq, states, rewards] = ...
    myhmmgenerate(num_trails, guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro ,1, []);

[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] =...
    myhmmtrain(emission_seq, envtype , rewards ,guess_trans_reward ,guess_trans_noreward, ...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',true, 'maxiterations', 1500);

[est_trans, est_emits] = hmmtrain(emission_seq , est_trans_noreward, guess_emit_homo,'VERBOSE',false, 'maxiterations', 1500);

diff_trans = est_trans_noreward - est_trans;
diff_emits = est_emits_homo - est_emits;
if (all(diff_trans(:)<tol) && all(diff_emits(:)< tol))
    fprintf('Test 2 - Pass\n')
else
    fprintf('Test 2 - Fail\n')
end

end

function tot_error = test3(guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro, num_trails)
%TEST 3 - generate a sequence of trials using myhmmgenerate with two environment type 
% but no rewarded states. The estimated transition and 
% emmits probabilities should be the same as the sequence generation parameters. 
% The guess in this test is the same as the equence generation parametrres.

[envtype,emission_seq, states, rewards] = ...
    myhmmgenerate(num_trails, guess_trans_reward, guess_trans_noreward, ...
    guess_emit_homo, guess_emit_hetro ,0.5, []);
[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = ...
    myhmmtrain(emission_seq, envtype , rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_emit_homo, guess_emit_hetro, 'VERBOSE',true, 'maxiterations', 1500);

tol = 0.1;
diff_trans = est_trans_noreward - guess_trans_noreward;
diff_emits_homo = est_emits_homo - guess_emit_homo;
diff_emits_hetro = est_emits_hetro - guess_emit_hetro;
if (all(diff_trans(:)<tol) && all(diff_emits_homo(:)< tol) && all(diff_emits_hetro(:)< tol))
    fprintf('Test 3 - Pass\n')
else
    fprintf('Test 3 - Fail\n')
end
tot_error = sum(abs(diff_trans(:)))+sum(abs(diff_emits_homo(:)))+sum(abs(diff_emits_hetro(:)));
end
 