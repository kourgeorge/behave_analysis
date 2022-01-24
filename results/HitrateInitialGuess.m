function HitrateInitialGuess()

hitrate = [];
repeats = 25;

types = repmat({'$N(\theta,0)$', '$N(\delta,1)$','$N(\Pi,1)$', '$N(\theta,1)$'},repeats,1);

for i=1:repeats
    [hitrate(i,1), ~] = runPerfectGuessExp(0,@noiseCAHMM);
    [hitrate(i,2), ~] = runPerfectGuessExp(1,@noiseTr);
    [hitrate(i,3), ~] = runPerfectGuessExp(1,@noisePolices);
    [hitrate(i,4), ~] = runPerfectGuessExp(1,@noiseCAHMM);
end

plot_gramm(hitrate(:),types(:));

end


function [hitrate, numIterations] = runPerfectGuessExp(noiseVec, noisefun)

theta_gt = getConstantGTparameters();

env_type_frac = 0.5;
sequenceLength = 1000;
testsequenceLength = 100;

[trainseq.envtype,trainseq.emissions, ~, trainseq.rewards] = ...
    mazehmmgenerate(sequenceLength, theta_gt.trR, theta_gt.trNR, ...
    theta_gt.eH, theta_gt.eT ,env_type_frac, [1 0; 0 1]);

[testseq.envtype,testseq.emissions, testseq.states, testseq.rewards] = ...
    mazehmmgenerate(testsequenceLength, theta_gt.trR, theta_gt.trNR, ...
    theta_gt.eH, theta_gt.eT ,env_type_frac, [1 0; 0 1]);

max_iterations = 500;
tolerance = 1e-4;

numIterations = [];
hitrate = [];


for noise=noiseVec
    
    theta_guess = noisefun(theta_gt, noise);
    
    [theta_hat.trR, theta_hat.trNR, theta_hat.eH, theta_hat.eT, logliks] = ...
    mazehmmtrain(trainseq.emissions, trainseq.envtype , trainseq.rewards ,theta_guess.trR ,theta_guess.trNR ,...
    theta_guess.eH, theta_guess.eT, 'VERBOSE',false, 'maxiterations', max_iterations, 'TOLERANCE', tolerance);

    estimated_policies_seq = mazehmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
            theta_hat.trR, theta_hat.trNR, theta_hat.eH, theta_hat.eT);
    
    
    policy_hit_rate = mean(MatchingPoliciesHitrate(estimated_policies_seq, testseq, theta_hat, theta_gt));
    
    numIterations = [numIterations,length(logliks)];
    hitrate = [hitrate, policy_hit_rate];
end
end


function noised_theta = noiseCAHMM(theta, eps)
noised_theta = noiseTr(theta, eps);
noised_theta = noisePolices(noised_theta, eps);
end

function noised_theta = noiseTr(theta, eps)
noised_theta.trR  = NoiseProbabilityMatrix(eps, theta.trR);
noised_theta.trNR = NoiseProbabilityMatrix(eps, theta.trNR);
noised_theta.eH = theta.eH;
noised_theta.eT = theta.eT;
end

function noised_theta = noisePolices(theta, eps)
noised_theta.trR  = theta.trR;
noised_theta.trNR = theta.trNR;
noised_theta.eH = NoiseProbabilityMatrix(eps, theta.eH);
noised_theta.eT = NoiseProbabilityMatrix(eps, theta.eT);
end

function plot_gramm(hitrate, types)

clear g;
x = repmat({' ',' ', ' ',' '},length(hitrate)/4,1);
g(1,1)=gramm('x', x(:) ,'y',hitrate,'color',types);

g(1,1).stat_boxplot('dodge',1, 'width',0.8)

g.set_names('x','','y','$hitrate$','color','');
g.set_text_options('font','Helvetica',...
    'base_size',18,...
    'label_scaling',1.2,...
    'legend_scaling',1,...
    'legend_title_scaling',1,...
    'facet_scaling',1,...
    'title_scaling',1, ...
    'Interpreter', 'latex');
g.axe_property('YLim',[0 1]);
g.set_order_options('color',1);
figure()
g.coord_flip();
g.draw();
g.export('file_name','hitrate','file_type','png')

end
