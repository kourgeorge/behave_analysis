function CAhitrate_hmmvsmazehmm()

training_seq_lengths = floor(linspace(50,1300,25));
repetitions = 25;

models = {'$\hat{\theta}_{HMM}$', '$\hat{\theta}_{CA-HMM}$', '$\theta_{CA-HMM}$'};

hitrates_rep = [];
for rep=1:repetitions
    hitrates_rep(:,:,rep) = calcPdiv(training_seq_lengths);
end
    
%gramm_graph(hitrates, models, training_seq_lengths);

show_graph(hitrates_rep, training_seq_lengths, models);
end

function show_graph(hitrates, training_seq_lengths, labels)
hitrates = permute(hitrates, [3 1 2]);
% hitrates: dim 1 - repetitions
%           dim 2 - sequence length
%           dim 3 - type - hmm, mazehmm, mazehmm_gt

colors = [0.8500 0.3250 0.0980;
    77/255,175/255,74/255;
    0 0.4470 0.7410];

figure;
hold on;
h={};
set(gca,'fontsize',18)
for i=1:size(hitrates,3)
    h(i)={shadedErrorBar(training_seq_lengths,hitrates(:,:,i),{@mean,@sem}, {'-','Color',colors(i,:), 'LineWidth',2},3)};
end

legend([h{1}.mainLine,h{2}.mainLine,h{3}.mainLine],[labels(1),labels(2),labels(3)], 'Interpreter','latex')

xlabel('$\ell^{train}$','Interpreter','latex', 'FontSize', 25)
ylabel('$Pdiv(z,\hat{z}; \theta,\hat{\theta)}$','Interpreter','latex', 'FontSize', 25)

hold off;

end

function [trainseq, testseq] = createTrainAndTestSequences(train_len, test_len, theta_gt, env_type_frac, R)
    
    [trainseq.envtype, trainseq.emissions, trainseq.states, trainseq.rewards] = ...
        mazehmmgenerate(train_len, theta_gt.trR, theta_gt.trNR, ...
        theta_gt.eH, theta_gt.eT ,env_type_frac, R);

    [testseq.envtype, testseq.emissions, testseq.states, testseq.rewards] = ...
        mazehmmgenerate(test_len, theta_gt.trR, theta_gt.trNR, ...
        theta_gt.eH, theta_gt.eT ,env_type_frac, R);
    
end

function hitrates = calcPdiv(training_seq_lengths)

theta_guess.trR = getrandomdistribution(4,4);
theta_guess.trNR = getrandomdistribution(4,4);
theta_guess.eH = getrandomdistribution(4,2);
theta_guess.eT = getrandomdistribution(4,2);

max_iter = 200;
tolerance = 1e-4;


theta_gt = getGTparameters(4);

%CAHHM using GT
% Neurolate and then tabulate GT policices to make it comparabale with
% the estimated CAHMM.
% theta_gt_neural=theta_gt;
gt_policies = neurolate_tabular_policies({theta_gt.eH ,theta_gt.eT});
% table_policies = tabulate_neural_policies(gt_policies, 2);
% theta_gt_neural.eH = table_policies{1};
% theta_gt_neural.eT = table_policies{2};
    
hitrates=[];
for seq_length=training_seq_lengths
    [trainseq, testseq] = createTrainAndTestSequences(seq_length, 100, theta_gt, 0.5, [0 1; 1 0]);
    %HMM%
    %Train
    [theta_hmm.trR, theta_hmm.eH] = ...
        hmmtrain(trainseq.emissions, theta_guess.trR, theta_guess.eH, 'VERBOSE',false, 'maxiterations', max_iter, 'tolerance', tolerance);
    theta_hmm.trNR = theta_hmm.trR;
    theta_hmm.eT = theta_hmm.eH;
    %Estimate
    estimatedstates_hmm = hmmviterbi(testseq.emissions,theta_hmm.trR,theta_hmm.eH);
    
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
    
    
    estimatedstates_gt = scahmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
        theta_gt.trR, theta_gt.trNR, gt_policies);
    
    cahmm_gt_hitrate = mean(PolicyDivergence(estimatedstates_gt, testseq, theta_gt, theta_gt));
    cahmm_hitrate = mean(PolicyDivergence(estimatedstates_cahmm, testseq, theta_cahmm, theta_gt));
    hmm_hitrate = mean(PolicyDivergence(estimatedstates_hmm, testseq, theta_hmm, theta_gt));
    hitrates = [hitrates; hmm_hitrate, cahmm_hitrate, cahmm_gt_hitrate];
    
end
end

function gramm_graph(hitrates, models, training_seq_lengths)

hit = [];
len = [];
mod = [];
for r=1:size(hitrates, 1) %repetitions
    for l=1:length(training_seq_lengths)
        for m=1:length(models)
            hit = [hit; hitrates(l,m,r)];
            len = [len; {num2str(training_seq_lengths(l))}];
            mod = [mod; models(m)];
        end
    end
end

clear g;
g(1,1)=gramm('x',len,'y',hit,'color',mod);

%g(1,1).stat_violin('fill','transparent');
g(1,1).stat_boxplot()

g.set_names('x','$\ell^{train}$','y','$hitrate$','color',' ');
g.set_text_options('font','Helvetica',...
    'base_size',18,...
    'label_scaling',1.2,...
    'legend_scaling',1,...
    'legend_title_scaling',1,...
    'facet_scaling',1,...
    'title_scaling',1, ...
    'Interpreter', 'latex');


figure()
g.coord_flip();
g.draw();

end

