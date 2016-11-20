function test_hmmgenerate()



%test_basics();
test2();

end


function test_basics()
% create the true transition and emission matrix.
eps = 0.1;
eH = [1-eps eps ; 1-eps eps; 1-eps eps];
eT = [eps 1-eps; eps 1-eps; eps 1-eps];

trR = [0.8 0.1 0.1;
    0.1 0.8 0.1;
    0.1 0.1 0.8];
    
trNR = [0.2 0.4 0.4;
    0.4 0.2 0.4;
    0.4 0.4 0.2];

% generate sequence
num_trials = 10000;
[envtype,seq,~,~] = mazehmmgenerate(num_trials, trR, trNR, eH, eT, 0.5, [1 0; 0 1] );

% check envtypefrac
frac_env1 = length(find(envtype==1))/num_trials;
if (frac_env1<0.51 && frac_env1>0.49)
    disp('Pass')
else
    disp('Fail')
end

% check the probability of the emit for the homo environemnt configuration
emits_env1 = seq(envtype==1);
emits1_env1 = find(emits_env1==1);
emits1_env1_frac = length(emits1_env1)/length(emits_env1);
if (emits1_env1_frac<0.91 && emits1_env1_frac>0.89)
    disp('Pass')
else
    disp('Fail')
end

end


function test2()
%compare the performnce of hmmtrain on both hmmgenerate and
%mazehmmgenerate. If hmmgenerate is correct hmm train will do the same 
%on the both sequences.

eps = 0.1;
eH = [1-eps eps ; 1-eps eps; 1-eps eps];
eT = [eps 1-eps; eps 1-eps; eps 1-eps];

trR = [0.8 0.1 0.1;
    0.1 0.8 0.1;
    0.1 0.1 0.8];
    
trNR = [0.2 0.4 0.4;
    0.4 0.2 0.4;
    0.4 0.4 0.2];

num_trials = 500;
[~,mazeseq,~,~] = mazehmmgenerate(num_trials, trR, trNR, eH, eT, 1, [0 0; 0 0] );

[seq, ~] = hmmgenerate(num_trials, trNR, eH);

guess_tr = rand(3);
guess_e = rand(3,2);

[guessTR,guessE,logliks] = hmmtrain(mazeseq, guess_tr, guess_e, 'maxiterations',500);

[guessTR2,guessE2,logliks2] = hmmtrain(seq, guess_tr, guess_e, 'maxiterations', 500);
end