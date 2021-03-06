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

       
%testSingleRewardedEnvHMMGenerate(tr, e);
%testNoRewardedEnvMazeHMMGenerate(tr, e);
%testNoRewardedEnvMazeHMMGenerate(tr, e);
%test4();
%compare_mazehmm_and_hmm_core_exp();
%compare_mazehmm_and_hmm_random_noreward();
compare_mazehmm_and_hmm_random_reward();
%compare_mazehmm_and_hmm_random_both();
%testPerfectGuess()

end

function testSingleRewardedEnvHMMGenerate(tr, e)
% TEST 1 - create sequence of a single environemnt type (h) 
% and reward and make sure the reduction of the
% generalized implementation (mazehmmtrain) emits the same results as the original
% hmm. 
% generate the sequnce using hmmgenerate (no myhmmgenrate).
num_trails = 500;
envtype = ones(1, num_trails); 
rewards = zeros(1, num_trails);
[emission_seq, ~] = hmmgenerate(num_trails,tr,e);
guess_tr = rand(5);
guess_e = rand(5,2);
[~, est_tr_mazehmm , est_e_mazehmm, ~] = ...
    mazehmmtrain(emission_seq, envtype , rewards ,guess_tr ,guess_tr ...
    ,guess_e, guess_e,'VERBOSE',false, 'maxiterations', 1500);
[est_tr_hmm, est_e_hmm] = hmmtrain(emission_seq , guess_tr, guess_e,'VERBOSE',false, 'maxiterations', 1500);

if (all(all(est_tr_mazehmm ==  est_tr_hmm)) && all(all(est_e_mazehmm == est_e_hmm)))
    fprintf('Test 1 - Pass\n')
else
    fprintf('Test 1 - Fail\n')
end

end

function testSingleRewardedEnvMazeHMMGenerate(tr, e)
% TEST 2 - generate data using mazehmmgeneratedata. The data is only homo 
% environment and no reward. Compare hmmtrain and mazehmmtrain. The
% difference in the transition probabilitis and the emission probability
% estimation should be identical.

num_trails = 500;
% on synthetic data with no rewarded states and only homo env type.
[envtype,emission_seq, ~, rewards] = ...
    mazehmmgenerate(num_trails, tr, tr, e, e ,1, [0 0;0 0;]);

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

function testNoRewardedEnvMazeHMMGenerate(tr, e)
%TEST 3 - generate a sequence of trials using mazehmmgenerate with two environment type 
% but no rewarded states. The estimated transition and 
% emmits probabilities should be the same as the sequence generation parameters. 
% The guess in this test is the same as the equence generation parametrres.

num_trails = 100;

guess_tr = rand(5);
guess_e = rand(5,2);

[envtype, seq, ~, rewards] = ...
    mazehmmgenerate(num_trails, tr, tr, e, e ,0.5, [0 0; 0 0]);
[est_trans_reward, est_trans_noreward, est_emits_homo, est_emits_hetro] = ...
    mazehmmtrain(seq, envtype , rewards ,guess_tr ,guess_tr ,...
    guess_e, guess_e, 'VERBOSE', false, 'maxiterations', 500);


[est_tr_hmm, est_e_hmm] = hmmtrain(seq, guess_tr, guess_e, 'maxiterations', 500);
tol = 0.2;
diff_trans = est_trans_noreward - est_tr_hmm;
diff_emits_hetro = est_emits_hetro - est_e_hmm;
if (all(diff_trans(:)<tol)  && all(diff_emits_hetro(:)< tol))
    fprintf('Test 3 - Pass\n')
else
    fprintf('Test 3 - Fail\n')
end
end
 
function test4()
% make sure the emits matrices of the two configuration is converging to
% the right point, in non reward configuration.

eH = [0.1 0.9;
    0.9 0.1];
eH_guess = [0.3 0.7;
        0.7 0.3];

eT = [0.9 0.1;
    0.1 0.9]; 
eT_guess = [0.7 0.3;
    0.3 0.7]; 
         
trNR = [0.1 0.9 
    0.9 0.1];
trR = randprobmatrix(2,2);

guess_tr = trNR;

num_trials = 10000;
[envtype,seq,states,rewards] = mazehmmgenerate(num_trials, trR, trNR, eH, eT, 0.5, [0 0; 0 0] );


[est_trR_mazehmm,est_trNR_mazehmm,est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
    mazehmmtrain(seq, envtype , rewards ,guess_tr ,guess_tr, ...
    eH_guess, eT_guess, 'VERBOSE', false, 'maxiterations', 500);

if (mean(KLDiv(est_eHomo_mazehmm,eH))<0.2 && mean(KLDiv(est_eHetro_mazehmm,eT))<0.2)
    disp('Pass')
else
    disp('Fail')
end

end

function compare_mazehmm_and_hmm_core_exp()
%This is a well designed outputs to emulate a possible situation in boch
%the majority of the outputs are 2 but the emission matrix is [0 1; 1 0]
%and [1 0; 0 1] and not [0 1; 0 1] which will the regular hmm will
%estimate.

% so basically this experiment was designed to "confuse" hmm and smake sure
% that the mazehmm is doing much better.

% 1. Generate a sequence of outputs so that vast majority of the output
% are 2 but the proces generating those outputs is transitioning between
% state q and state 2 with 
% 2. Train hmm based on the observation sequnce
% 3. Train mazeHmm on the observations
% 4. Check the difference in the probability matrices and the latent
% states.


eH = [0.1 0.9;
    0.9 0.1];

eT = [0.9 0.1;
    0.1 0.9]; 
         
trNR = [0.1 0.9 
    0.9 0.1];


num_trials=200; %half the total length)
A=[ones(1,num_trials/2); 2*ones(1,num_trials/2)];
A=A(:);

envtype = A';
seq = 2* ones(1,num_trials);
rewards = zeros(1,num_trials);
guess_tr = [0.5 0.5;0.5 0.5];
eps = 0.001;
guess_eH = [.5 .5; .5 .5] + [-eps eps; eps -eps];
guess_eT = [.55 .45; .45 .55] + [eps -eps; -eps eps];

%extend the experimnts to include emission 1 for both env1 and env2
envtype = [envtype, ones(1,10)];
seq = [seq, [2 1 2 1 2 1 2 1 2 1]];
rewards = [rewards, zeros(1,10)];

envtype = [envtype, 2*ones(1,10)];
seq = [seq, [1 2 1 2 1 2 1 2 1 2]];
rewards = [rewards, zeros(1,10)];


[est_tr_hmm, est_e_hmm, logliks_hmm] = hmmtrain(seq, guess_tr, [.5 .5; .5 .5], 'maxiterations', 500, 'VERBOSE', true);


[est_trR_mazehmm, est_trNR_mazehmm, est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
    mazehmmtrain(seq, envtype , rewards ,guess_tr ,guess_tr, ...
    guess_eH, guess_eT, 'VERBOSE', true, 'maxiterations', 500, 'TOLERANCE',1e-6);


diff_tr_hmm = dist_prob_matrices(trNR, trNR, est_tr_hmm, est_tr_hmm);
diff_emits_hmm = dist_prob_matrices(eH, eT, est_e_hmm, est_e_hmm);
diff_hmm = diff_tr_hmm+diff_emits_hmm;

diff_tr_mazehmm = dist_prob_matrices(trNR, trNR, est_trNR_mazehmm, est_trNR_mazehmm);
diff_emits_mazehmm = dist_prob_matrices(eH, eT, est_eHomo_mazehmm, est_eHetro_mazehmm);
diff_mazehmm =  diff_tr_mazehmm+diff_emits_mazehmm;

if (diff_tr_hmm>diff_tr_mazehmm && diff_emits_hmm>diff_emits_mazehmm)
    disp('Pass')
else
    disp('Fail')
end
end

function compare_mazehmm_and_hmm_random_noreward()

% 1. Generate a sequence of outputs using hmmgenerate with no reward
% 2. Train hmm based on the observation sequence
% 3. Train mazeHmm on the observations
% 4. Check the difference in the probability matrices and the latent
% states.


eH = [0.1 0.9;
    0.9 0.1];

eT = [0.9 0.1;
    0.1 0.9]; 
         
trNR = [0.1 0.9 
    0.9 0.1];

eps = 0.1;
guess_eH = [.5 .5; .5 .5] + [-eps eps; eps -eps];
guess_eT = [.5 .5; .5 .5] + [eps -eps; -eps eps];
guess_tr = [0.5 0.5;0.5 0.5];

num_trials = 100;
[envtype,seq,states,rewards] = mazehmmgenerate(num_trials, trNR, trNR, eH, eT, 0.5, [0 0; 0 0] );


[est_tr_hmm, est_e_hmm, logliks_hmm] = hmmtrain(seq, guess_tr, [.5 .5; .5 .5], 'maxiterations', 500, 'VERBOSE', true);


[est_trR_mazehmm, est_trNR_mazehmm, est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
    mazehmmtrain(seq, envtype , rewards ,guess_tr ,guess_tr, ...
    guess_eH, guess_eT, 'VERBOSE', true, 'maxiterations', 500);


diff_tr_hmm = dist_prob_matrices(trNR, trNR, est_tr_hmm, est_tr_hmm);
diff_emits_hmm = dist_prob_matrices(eH, eT, est_e_hmm, est_e_hmm);
diff_hmm = diff_tr_hmm+diff_emits_hmm;

diff_tr_mazehmm = dist_prob_matrices(trNR, trNR, est_trNR_mazehmm, est_trNR_mazehmm);
diff_emits_mazehmm = dist_prob_matrices(eH, eT, est_eHomo_mazehmm, est_eHetro_mazehmm);
diff_mazehmm =  diff_tr_mazehmm+diff_emits_mazehmm;

if (diff_tr_hmm>diff_tr_mazehmm && diff_emits_hmm>diff_emits_mazehmm)
    disp('Pass')
else
    disp('Fail')
end
end

function compare_mazehmm_and_hmm_random_reward()

% 1. Generate a sequence of outputs using hmmgenerate with reward
% 2. Train hmm based on the observation sequence
% 3. Train mazeHmm on the observations
% 4. Check the difference in the probability matrices and the latent states.


eH = [0.1 0.9;
    0.9 0.1];

eT = [0.9 0.1;
    0.1 0.9]; 
         
trR = [0.9 0.1 
    0.1 0.9];

[guess_eH,guess_eT] = createGuessProbabilityMatrices(eH, eT, 0.5);
guess_tr = createGuessProbabilityMatrices(trR, trR, 0.5);

num_trials = 1000;
tolerance = 1e-4;
[envtype,seq,states,rewards] = mazehmmgenerate(num_trials, trR, trR, eH, eT, 0.5, [1 1; 1 1] );


[est_tr_hmm, est_e_hmm, logliks_hmm] = hmmtrain(seq, guess_tr, [.5, .5;.5 .5 ] , 'maxiterations', 500, 'VERBOSE', true, 'tolerance', tolerance);


[est_trR_mazehmm, est_trNR_mazehmm, est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
    mazehmmtrain(seq, envtype , rewards ,trR ,guess_tr, ...
    [.5, .5;.5 .5 ], [.5, .5;.5 .5 ], 'VERBOSE', true, 'maxiterations', 500, 'tolerance', tolerance);


diff_tr_hmm = dist_prob_matrices(trR, trR, est_tr_hmm, est_tr_hmm);
diff_emits_hmm = dist_prob_matrices(eH, eT, est_e_hmm, est_e_hmm);
diff_hmm = diff_tr_hmm+diff_emits_hmm;

diff_tr_mazehmm = dist_prob_matrices(trR, trR, est_trR_mazehmm, est_trR_mazehmm);
diff_emits_mazehmm = dist_prob_matrices(eH, eT, est_eHomo_mazehmm, est_eHetro_mazehmm);
diff_mazehmm =  diff_tr_mazehmm+diff_emits_mazehmm;

if (diff_tr_hmm>diff_tr_mazehmm && diff_emits_hmm>diff_emits_mazehmm)
    disp('Pass')
else
    disp('Fail')
end
end

function compare_mazehmm_and_hmm_random_both()

% 1. Generate a sequence of outputs using hmmgenerate with reward
% 2. Train hmm based on the observation sequnce
% 3. Train mazeHmm on the onservations
% 4. Check the difference in the probability matrices and the latent
% states.


eH = [0.1 0.9;
    0.9 0.1];

eT = [0.9 0.1;
    0.1 0.9]; 
         
trNR = [0.1 0.9 
    0.9 0.1];


trR = [0.9 0.1 
    0.1 0.9];

eps = 0.2;
guess_eH = [.5 .5; .5 .5] + [-eps eps; eps -eps];
guess_eT = [.5 .5; .5 .5] + [eps -eps; -eps eps];
guess_tr = [0.5 0.5;0.5 0.5];

num_trials = 5;
[envtype,seq,states,rewards] = mazehmmgenerate(num_trials, trR, trR, eH, eT, 0.5, [1 0; 0 1] );


[est_tr_hmm, est_e_hmm, logliks_hmm] = hmmtrain(seq, guess_tr, [.5 .5; .5 .5], 'maxiterations', 500, 'VERBOSE', true);


[est_trR_mazehmm, est_trNR_mazehmm, est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
    mazehmmtrain(seq, envtype , rewards ,guess_tr ,guess_tr, ...
    guess_eH, guess_eT, 'VERBOSE', true, 'maxiterations', 500);


diff_tr_hmm = dist_prob_matrices(trR, trR, est_tr_hmm, est_tr_hmm);
diff_emits_hmm = dist_prob_matrices(eH, eT, est_e_hmm, est_e_hmm);
diff_hmm = diff_tr_hmm+diff_emits_hmm;

diff_tr_mazehmm = dist_prob_matrices(trR, trNR, est_trR_mazehmm, est_trNR_mazehmm);
diff_emits_mazehmm = dist_prob_matrices(eH, eT, est_eHomo_mazehmm, est_eHetro_mazehmm);
diff_mazehmm =  diff_tr_mazehmm+diff_emits_mazehmm;

if (diff_tr_hmm>diff_tr_mazehmm && diff_emits_hmm>diff_emits_mazehmm)
    disp('Pass')
else
    disp('Fail')
end
end

function testPerfectGuess()
%the algorithm should converge on 1 or 2 iterations and the estimations
%should be almost perfect.

eH = [0.1 0.9;
    0.9 0.1];

eT = [0.9 0.1;
    0.1 0.9]; 
         
trNR = [0.1 0.9 
    0.9 0.1];

trR = [0.9 0.1 
    0.1 0.9];


num_trials = 1000;
[envtype, seq, ~, rewards] = mazehmmgenerate(num_trials, trR, trNR, eH, eT, 0.5, [1 0; 0 1] );

[est_trR_mazehmm, est_trNR_mazehmm, est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
    mazehmmtrain(seq, envtype , rewards ,trR ,trNR, ...
    eH, eT, 'VERBOSE', true, 'maxiterations', 500, 'TOLERANCE',1e-2);

diff_tr_mazehmm = dist_prob_matrices(trR, trNR, est_trR_mazehmm, est_trNR_mazehmm);

if (diff_tr_mazehmm < 0.1)
    disp('Pass')
else
    disp('Fail')
end

end

%%%%%%%%%% Helper Functions %%%%%%%%%
function dist = dist_prob_matrices(true1, true2, est1, est2)

%dist = mean((KLDiv(true1,est1)) + mean(KLDiv(true2,est2))) /2;
dist = mean(norm(true1-est1)+norm(true2-est2))/2;
end