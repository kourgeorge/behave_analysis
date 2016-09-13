function test_mazehmmtrain()

eps = 0.05;
e = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps;
             0.5 0.5];
         

tr = [0.6 0.25 0.05 0.05 0.05;
           0.25 0.6 0.05 0.05 0.05;
           0.05 0.05 0.6 0.25 0.05;
           0.05 0.05 0.25 0.6 0.05
           0.2 0.2 0.2 0.2 0.2];

       
%test1(tr, e);
%test2(tr, e);
test3(tr, e)

end

function test1(tr, e)
% TEST 1 - create sequence of a single environemnt type (h) 
% and reward and make sure the reduction of the
% generalized implementation (mazehmmtrain) emits the same results as the original
% hmm. 
% generate the sequnce using hmmgenerate (no myhmmgenrate).
num_trails = 500;
envtype = ones(1, num_trails); 
rewards = ones(1, num_trails);
[emission_seq, ~] = hmmgenerate(num_trails,tr,e);
guess_tr = rand(5);
guess_e = rand(5,2);
[est_tr_mazehmm, ~ , est_e_mazehmm, ~] = ...
    mazehmmtrain(emission_seq, envtype , rewards ,guess_tr ,guess_tr ...
    ,guess_e, guess_e,'VERBOSE',false, 'maxiterations', 1500);
[est_tr_hmm, est_e_hmm] = hmmtrain(emission_seq , guess_tr, guess_e,'VERBOSE',false, 'maxiterations', 1500);

if (all(all(est_tr_mazehmm ==  est_tr_hmm)) && all(all(est_e_mazehmm == est_e_hmm)))
    fprintf('Test 1 - Pass\n')
else
    fprintf('Test 1 - Fail\n')
end

end

function test2(tr, e)
% TEST 2 - generate data using mazehmmgeneratedata. The data is only homo 
% environment and no reward. Compare hmmtrain and mazehmmtrain. The
% difference in the transition probabilitis and the emission probability
% estimation should be identical.

num_trails = 500;
% on synthetic data with no rewarded states and only homo env type.
[envtype,emission_seq, ~, rewards] = ...
    mazehmmgenerate(num_trails, tr, tr, e, e ,1, []);

guess_tr = rand(5);
guess_e = rand(5,2);

[~, est_tr_mazehmm, est_e_mazehmm, ~] =...
    mazehmmtrain(emission_seq, envtype , rewards ,guess_tr ,guess_tr, ...
    guess_e, guess_e, 'VERBOSE', false, 'maxiterations', 1500);

[est_tr_hmm, est_e_hmm] = hmmtrain(emission_seq , guess_tr, guess_e,'VERBOSE',false, 'maxiterations', 1500);


if (all(all(est_tr_mazehmm ==  est_tr_hmm)) && all(all(est_e_mazehmm == est_e_hmm)))
    fprintf('Test 2 - Pass\n')
else
    fprintf('Test 2 - Fail\n')
end

end

function test3(tr, e)
%TEST 3 - generate a sequence of trials using mazehmmgenerate with two environment type 
% but no rewarded states. The estimated transition and 
% emmits probabilities should be the same as the sequence generation parameters. 
% The guess in this test is the same as the equence generation parametrres.

num_trails = 10000;

guess_tr = rand(5);
guess_e = rand(5,2);

[envtype, seq, ~, rewards] = ...
    mazehmmgenerate(num_trails, tr, tr, e, e ,0.5, []);
[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = ...
    mazehmmtrain(seq, envtype , rewards ,guess_tr ,guess_tr ,...
    guess_e, guess_e, 'VERBOSE', false, 'maxiterations', 1500);


[est_tr_hmm, est_e_hmm] = hmmtrain(seq, guess_tr, guess_e, 'maxiterations', 1500);
tol = 0.2;
diff_trans = est_trans_noreward - est_tr_hmm;
diff_emits_hetro = est_emits_hetro - est_e_hmm;
if (all(diff_trans(:)<tol)  && all(diff_emits_hetro(:)< tol))
    fprintf('Test 3 - Pass\n')
else
    fprintf('Test 3 - Fail\n')
end
end
 