function poster_comparison( )
%POSTER_COMPARISON Summary of this function goes here
%   Detailed explanation goes here

e1 = [0.9 0.1;
    0.1 0.9
    0.9 0.1
    0.1 0.9];

e2 = ones(4,2)-e1;
 
trNR = [0.1 0.3 0.3 0.3 
    0.3 0.1 0.3 0.3 
    0.3 0.3 0.1 0.3 
    0.3 0.3 0.3 0.1];

trR = [0.8 0 0.1 0.1 
    0 0.8 0.1 0.1
    0.1 0.1 0.8 0
    0.1 0.1 0 0.8];


tr_acc=[];
emits_acc=[];

for i=1:200
    
    guess_eH = 0.5*abs(randn(4,2));
    guess_eT = 0.5*abs(randn(4,2));
    guess_trnr = 0.25*abs(randn(4));
    guess_trr = 0.25*abs(randn(4));
  
    
    
    num_trials = 100;
    [envtype,seq,states,rewards] = mazehmmgenerate(num_trials, trR, trNR, e1, e2, 0.4, [1 0; 0 1] );
    
    [est_tr_hmm, est_e_hmm, logliks_hmm] = hmmtrain(seq, guess_trnr, guess_eH, 'maxiterations', 500, 'VERBOSE', false);
    
    
    [est_trR_mazehmm, est_trNR_mazehmm, est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
        mazehmmtrain(seq, envtype , rewards ,guess_trr ,guess_trnr, ...
        guess_eH, guess_eT, 'VERBOSE', true, 'maxiterations', 500);
    
    
    diff_tr_hmm = dist_prob_matrices(trNR, trNR, est_tr_hmm, est_tr_hmm);
    diff_emits_hmm = dist_prob_matrices(e1, e2, est_e_hmm, est_e_hmm);
    
    
    diff_tr_mazehmm = dist_prob_matrices(trR, trNR, est_trR_mazehmm, est_trNR_mazehmm);
    diff_emits_mazehmm = dist_prob_matrices(e1, e2, est_eHomo_mazehmm, est_eHetro_mazehmm);
    
    
    emits_acc=[emits_acc;diff_emits_hmm,diff_emits_mazehmm];
    tr_acc = [tr_acc;diff_tr_hmm,diff_tr_mazehmm];
end

mean(tr_acc)
mean(emits_acc)

var(tr_acc)
var(emits_acc)

end

function dist = dist_prob_matrices(true1, true2, est1, est2)

%dist = mean((KLDiv(true1,est1)) + mean(KLDiv(true2,est2))) /2;
dist = mean(norm(true1-est1)+norm(true2-est2))/2;
end
