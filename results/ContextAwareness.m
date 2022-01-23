function contextAwareness
%CONTEXTAWARE Summary of this function goes here
%   Detailed explanation goes here

types = {'$\hat{\theta}_{HMM}$', '$\hat{\theta}_{CA-HMM}$', '$\theta_{CA-HMM}$'};
repetitions = 50;
[hitrates, gt_awarness] = calcPdiv(repetitions);

models = repmat(types,repetitions,1);
awarness = repmat(gt_awarness',1,length(types));
gramm_graph(awarness(:), hitrates(:), models(:));

end

function [hitrates, gt_awarness] = calcPdiv(repetitions)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.
theta_guess.trR = getrandomdistribution(2,2);
theta_guess.trNR = getrandomdistribution(2,2);
theta_guess.eH = getrandomdistribution(2,2);
theta_guess.eT = getrandomdistribution(2,2);

R = [1 0; 0 1];
train_len = 1000;
test_len = 100;
max_iter = 200;
tolerance = 1e-4;
gt_awarness= [];
hitrates=[];
for rep=1:repetitions
    theta_gt = get2policiesparameters(2);

    
    [trainseq.envtype, trainseq.emissions, trainseq.states, trainseq.rewards] = ...
        mazehmmgenerate(train_len, theta_gt.trR, theta_gt.trNR, ...
        theta_gt.eH, theta_gt.eT ,0.5, R);
    
    [testseq.envtype, testseq.emissions, testseq.states, testseq.rewards] = ...
        mazehmmgenerate(test_len, theta_gt.trR, theta_gt.trNR, ...
        theta_gt.eH, theta_gt.eT ,0.5, R);
    
      
    %HMM%
    %Train
    [theta_hmm.trR, theta_hmm.eH] = ...
        hmmtrain(trainseq.emissions, theta_guess.trR, theta_guess.eH, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);
    theta_hmm.trNR = theta_hmm.trR;
    theta_hmm.eT = theta_hmm.eH;
    %Estimate
    estimatedstates_hmm = hmmviterbi(testseq.emissions,theta_hmm.trR,theta_hmm.eH);
    
    %MazeHMM%
    %Train
    [theta_mazehmm.trR, theta_mazehmm.trNR, theta_mazehmm.eH, theta_mazehmm.eT] = ...
        mazehmmtrain(trainseq.emissions, trainseq.envtype , trainseq.rewards ,theta_guess.trR ,theta_guess.trNR ,...
        theta_guess.eH, theta_guess.eT, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);
    %Estimate
    try
        estimatedstates_mazehmm = mazehmmviterbi(testseq.emissions,testseq.envtype,testseq.rewards,...
            theta_mazehmm.trR,theta_mazehmm.trNR,theta_mazehmm.eH,theta_mazehmm.eT);
    catch exp
        estimatedstates_mazehmm = ones(1,length(testseq.emissions));
        warning(exp.message);
    end
    %mazeHM using GT
    %Estimate
    estimatedstates_gt = mazehmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
        theta_gt.trR, theta_gt.trNR, theta_gt.eH, theta_gt.eT);
    %%%%
    
    mazehmm_gt_hitrate= mean(ActionDivergence(estimatedstates_gt, testseq, theta_gt, theta_gt));
    mazehmm_hitrate = mean(ActionDivergence(estimatedstates_mazehmm, testseq, theta_mazehmm,  theta_gt));
    hmm_hitrate = mean(ActionDivergence(estimatedstates_hmm, testseq, theta_hmm, theta_gt));
    hitrates = [hitrates; hmm_hitrate, mazehmm_hitrate, mazehmm_gt_hitrate];
    gt_awarness(rep) = PoliciesContextAwareness(theta_gt);
end
end

function average_awarness = PoliciesContextAwareness(theta_gt)

policies = CreatePolicies({theta_gt.eH,theta_gt.eT});
awareness_score = [];
for i=1:length(policies)
    awareness_score(i) = PolicyContextAwareness (policies{i});
end

average_awarness=mean(awareness_score);
end

function awarness = PolicyContextAwareness(policy)
    awarness = JSDiv(policy(1,:), policy(2,:));
end


function gramm_graph(awarness, hitrates, models)

clear g;
g(1,1)=gramm('x',awarness,'y',hitrates(:),'color',models);

g.geom_point();
g(1,1).stat_glm()

g.set_names('x','$CA(\Pi)$','y','$Adiv$','color',' ');
g.set_text_options('font','Helvetica',...
    'base_size',18,...
    'label_scaling',1.2,...
    'legend_scaling',1,...
    'legend_title_scaling',1,...
    'facet_scaling',1,...
    'title_scaling',1, ...
    'Interpreter', 'latex');


figure();
g.set_order_options('color',0);
g.draw();

g.export('file_name','Adiv','file_type','png')
end