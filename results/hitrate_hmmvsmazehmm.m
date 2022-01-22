function hitrate_hmmvsmazehmm()

training_seq_lengths = floor(linspace(50,2000,50));
repetitions = 25;

models = {'$\hat{\theta}_{HMM}$', '$\hat{\theta}_{CA-HMM}$', '$\theta_{CA-HMM}$'};

hitrates = calcPdiv(training_seq_lengths, repetitions);

gramm_graph(hitrates, models, training_seq_lengths);

show_graph(hitrates, training_seq_lengths, models);
end

function show_graph(hitrates, x, labels)
hitrates = permute(hitrates, [3 1 2]);
% hitrates: dim 1 - repetitions
%           dim 2 - sequence length
%           dim 3 - type - hmm, mazehmm, mazehmm_gt



colors = [0 0.4470 0.7410;
    0.8500 0.3250 0.0980;
    0.9290 0.6940 0.1250];


figure;
hold on;
h={};
set(gca,'fontsize',18)
for i=1:size(hitrates,3)
    %errorbar(x,mean(hitrates(:,:,i)),sem(hitrates(:,:,i)), '-s','MarkerSize', 7, 'linewidth', 2, 'CapSize', 16) 
    h(i)={shadedErrorBar(x,hitrates(:,:,i),{@mean,@sem}, {'-','Color',colors(i,:), 'LineWidth',2},3)};
end

legend([h{1}.mainLine,h{2}.mainLine,h{3}.mainLine],[labels(1),labels(2),labels(3)], 'Interpreter','latex')

xlabel('$l^{train}$','Interpreter','latex', 'FontSize', 25)
ylabel('$Pdiv$','Interpreter','latex', 'FontSize', 25)

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

function hitrates_rep = calcPdiv(training_seq_lengths, repetitions)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.
theta_guess.trR = getrandomdistribution(4,4);
theta_guess.trNR = getrandomdistribution(4,4);
theta_guess.eH = getrandomdistribution(4,2);
theta_guess.eT = getrandomdistribution(4,2);

max_iter = 200;
tolerance = 1e-4;
hitrates = [];
hitrates_rep = [];
for rep=1:repetitions
    theta_gt = get_real_parameters(0.05);
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
        
        mazehmm_gt_hitrate= mean(MatchingPoliciesHitrate(estimatedstates_gt, testseq, theta_gt, theta_gt));
        mazehmm_hitrate = mean(MatchingPoliciesHitrate(estimatedstates_mazehmm, testseq, theta_mazehmm,  theta_gt));
        hmm_hitrate = mean(MatchingPoliciesHitrate(estimatedstates_hmm, testseq, theta_hmm, theta_gt));
        hitrates = [hitrates; hmm_hitrate, mazehmm_hitrate, mazehmm_gt_hitrate];
        
    end
    
    hitrates_rep = cat (3, hitrates_rep, hitrates);
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

g.set_names('x','$l^{train}$','y','$hitrate$','color',' ');
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

function theta_real = get_real_parameters(eps)
theta_real.eH = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

theta_real.eT = [1-eps eps; %o1l2 o2l1
    eps 1-eps;
    eps 1-eps;
    1-eps eps];


theta_real.trR = ...
    [0.7 0.1 0.1 0.1;
    0.1 0.7 0.1 0.1;
    0.1 0.1 0.7 0.1;
    0.1 0.1 0.1 0.7];


theta_real.trNR = ...
    [0.1 0.5 0.2 0.2;
    0.5 0.1 0.2 0.2;
    0.2 0.2 0.1 0.5;
    0.2 0.2 0.5 0.1];


end

