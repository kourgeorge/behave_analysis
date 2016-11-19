function [ output_args ] = poster_comparison( input_args )
%POSTER_COMPARISON Summary of this function goes here
%   Detailed explanation goes here

eH = [0.1 0.9;
    0.9 0.1];

eT = [0.9 0.1;
    0.1 0.9];

trNR = [0.1 0.9
    0.9 0.1];


tr_acc=[];
emits_acc=[];

for i=1:200
    
    eps = 0.5*randn();
    guess_eH = [.5 .5; .5 .5] + [-eps eps; eps -eps];
    guess_eT = [.5 .5; .5 .5] + [eps -eps; -eps eps];
    guess_tr = [0.5 0.5;0.5 0.5] + [eps -eps; -eps eps];
    
    guess_eH = eH + [-eps eps; eps -eps];
    guess_eT = eT + [eps -eps; -eps eps];
    guess_tr = trNR + [eps -eps; -eps eps];
    
    
    num_trials = 100;
    [envtype,seq,states,rewards] = mazehmmgenerate(num_trials, trNR, trNR, eH, eT, 0.5, [0 0; 0 0] );
    
    [est_tr_hmm, est_e_hmm, logliks_hmm] = hmmtrain(seq, guess_tr, [.5 .5; .5 .5], 'maxiterations', 500, 'VERBOSE', true);
    
    
    [est_trR_mazehmm, est_trNR_mazehmm, est_eHomo_mazehmm,est_eHetro_mazehmm,logliks_mazehmm] =...
        mazehmmtrain(seq, envtype , rewards ,guess_tr ,guess_tr, ...
        guess_eH, guess_eT, 'VERBOSE', true, 'maxiterations', 500);
    
    
    diff_tr_hmm = dist_prob_matrices(trNR, trNR, est_tr_hmm, est_tr_hmm);
    diff_emits_hmm = dist_prob_matrices(eH, eT, est_e_hmm, est_e_hmm);
    
    
    diff_tr_mazehmm = dist_prob_matrices(trNR, trNR, est_trNR_mazehmm, est_trNR_mazehmm);
    diff_emits_mazehmm = dist_prob_matrices(eH, eT, est_eHomo_mazehmm, est_eHetro_mazehmm);
    
    
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
