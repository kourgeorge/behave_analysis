function CAContextAwareness
%CONTEXTAWARE Summary of this function goes here
%   Detailed explanation goes here

types = {'$\hat{\theta}_{HMM}$', '$\hat{\theta}_{CA-HMM}$', '$\theta_{CA-HMM}$'};
repetitions = 100;
[hitrates, gt_awarness] = runExperiment(repetitions);

models = repmat(types,length(gt_awarness),1);
awarness = repmat(gt_awarness',1,length(types));
gramm_graph(awarness(:), hitrates(:), models(:));

end

function [hitrates, gt_awarness] = runExperiment(repetitions)
theta_guess.trR = getrandomdistribution(4,4);
theta_guess.trNR = getrandomdistribution(4,4);
theta_guess.eH = getrandomdistribution(4,2);
theta_guess.eT = getrandomdistribution(4,2);

R = [1 0; 0 1];
train_len = 1000;
test_len = 100;
max_iter = 200;
tolerance = 1e-4;
gt_awarness= [];
hitrates=[];
for rep=1:repetitions
    theta_gt = get2policiesparameters(4);
    %theta_gt = getGTparameters(4);
    
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
    
    %CA-HMM%
    %Train
    guess_policies = neurolate_tabular_policies({theta_guess.eH ,theta_guess.eT});
    
    [theta_cahmm.trR, theta_cahmm.trNR, policies_hat] = ...
        scahmmtrain(trainseq.emissions, trainseq.envtype , trainseq.rewards ,theta_guess.trR ,theta_guess.trNR ,...
        guess_policies, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);
    
    table_policies = tabulate_neural_policies(policies_hat, 2);

    theta_cahmm.eH = table_policies{1};
    theta_cahmm.eT = table_policies{2};

    
    %Estimate
    try
         estimatedstates_cahmm = scahmmviterbi(testseq.emissions,testseq.envtype,testseq.rewards,...
             theta_cahmm.trR,theta_cahmm.trNR,policies_hat);

    catch exp
        estimatedstates_cahmm = ones(1,length(testseq.emissions));
        warning(exp.message);
    end
    
    %CAHHM using GT
    gt_policies = neurolate_tabular_policies({theta_gt.eH ,theta_gt.eT}); 
    estimatedstates_gt = scahmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
        theta_gt.trR, theta_gt.trNR, gt_policies);
    
    
    cahmm_gt_hitrate= mean(PolicyDivergence(estimatedstates_gt, testseq, theta_gt, theta_gt));
    cahmm_hitrate = mean(PolicyDivergence(estimatedstates_cahmm, testseq, theta_cahmm,  theta_gt));
    hmm_hitrate = mean(PolicyDivergence(estimatedstates_hmm, testseq, theta_hmm, theta_gt));
    hitrates = [hitrates; hmm_hitrate, cahmm_hitrate, cahmm_gt_hitrate];
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

g.set_names('x','$CA(\Pi)$','y','$Pdiv(z^{test}, \hat{z}^{test}_t)$','color',' ');
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
g.set_layout_options('legend_pos',[0.17 0.65 0.1 0.3])
g.draw();

g.export('file_name','Adiv','file_type','png')
end