function mazehmmperfectguess()

hitrate = [];
steps = 5;
from = 0;
to = 1;
noiseVec = linspace(from, to, steps);
repeats = 25;
for i=1:repeats
    [hitrate(i,:,1), ~] = getdistanceforsequence(noiseVec,@noiseCAHMM);
    [hitrate(i,:,2), ~] = getdistanceforsequence(noiseVec,@noiseTr);
    [hitrate(i,:,3), ~] = getdistanceforsequence(noiseVec,@noisePolices);
end

plot_gramm(hitrate,noiseVec);
%hitrate_res = permute(hitrate_res, [1 3 2]);
colors = [0.6350 0.0780 0.1840;
        0.3010 0.7450 0.9330;
        0.4660 0.6740 0.1880];

figure;
set(gca,'fontsize',22)
hold on;
for j=1:3
    axe(j) = shadedErrorBar(linspace(from,to,steps),hitrate(:,:,j),{@mean,@sem}, {'-','Color',colors(j,:)},3);
end
xlabel('$\lambda$', 'Interpreter', 'latex')
ylabel('$hitrate$', 'Interpreter', 'latex')
set(gca,'fontsize',14)
box off;
yline(0.25, '--')
ylim([0.2,0.9]);
legend([axe(1).mainLine,axe(2).mainLine, axe(3).mainLine],...
    '$N(\theta,\lambda)$','$N(\delta,\lambda)$','$N(\Pi,\lambda)$', 'Interpreter','latex')
hold off;

end


function [hitrate, numIterations] = getdistanceforsequence(noiseVec, noisefun)

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

function plot_gramm(hitrate, noiseVec)

types = {'$N(\theta,\lambda)$','$N(\delta,\lambda)$','$N(\Pi,\lambda)$'};

types = {'Both','Transitions','Policies'};


hit = [];
lambda = [];
type = [];
for r=1:size(hitrate,1) %repetition
    for s=1:size(hitrate,2) %lambda - noise size
        for n=1:size(hitrate,3) % noised-param
            hit = [hit; hitrate(r,s,n)];
            lambda = [lambda; {num2str(noiseVec(s))}];
            if s == 1
                type = [type; types(1)];
            else
                type = [type; types(n)];
            end
        end
    end
end


clear g;

g(1,1)=gramm('x',type,'y',hit,'color',lambda);

g(1,1).stat_violin('fill','transparent');
%g(1,1).stat_boxplot()

g.set_names('x','','y','$hitrate$','color','$\lambda$');
g.set_text_options('font','Helvetica',...
    'base_size',18,...
    'label_scaling',1.2,...
    'legend_scaling',1,...
    'legend_title_scaling',1,...
    'facet_scaling',1,...
    'title_scaling',1, ...
    'Interpreter', 'latex');


figure()
%g.coord_flip();
g.draw();
end
