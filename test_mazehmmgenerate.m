
% create the true transition and emission matrix.
eps = 0.05;
e = [1-eps eps;
             eps 1-eps;
             1-eps eps;
             eps 1-eps;
             0.5 0.5];
         

tr = [0.8 0.05 0.05 0.05 0.05;
           0.05 0.8 0.05 0.05 0.05;
           0.05 0.05 0.8 0.05 0.05;
           0.05 0.05 0.05 0.8 0.05
           0.2 0.2 0.2 0.2 0.2];
       

lengths = linspace(10,10000, 15);

%craete the guess matrices.
tr_guess = rand(5);
e_guess = rand(5,2);

%create a sequence based on the the true tranisiotn and emission matrices.
[envtype,seq,states,rewards ] = mazehmmgenerate(10000, tr, tr, e, e, 1, []);

res = [norm(tr_guess-tr)+norm(e_guess-e)];

% train the model based on the generated data, with increasing lengthes of
% samples.
for seqlen=lengths   
    [guessTR,guessE,logliks] = hmmtrain(seq(1:seqlen),tr_guess, e_guess, 'maxiterations', 500);
    
    %[~, guessTR, guessE, ~] = ...
    %mazehmmtrain(seq(1:seqlen), envtype(1:seqlen) , rewards(1:seqlen) ,tr_guess ,tr_guess ,...
    %e_guess, e_guess, 'maxiterations', 500);

    res=[res, norm(guessTR-tr)+norm(guessE-e)];

end

% make sure that the error getting smaller.
plot([0,lengths],res)