function CAParameterRecovery()
%PARAMETERRECOVERY Summary of this function goes here
%   Detailed explanation goes here

transJS = [];
policiesJS = [];
policy_hit_rate_tr_noise = [];
policy_hit_rate_policy_noise = [];
policy_hit_rate_theta_noise = [];
repeats = 45;
noisevec = [0, 0.25 ,0.5, 0.75, 1, -1]; %minus is untrained model with given noise
hitrate = [];

for i=1:repeats
    [transJS(i,:), ~, policy_hit_rate_tr_noise(i,:)] = runPerfectGuessExp(noisevec,@noiseTr);
    [~, policiesJS(i,:), policy_hit_rate_policy_noise(i,:)] = runPerfectGuessExp(noisevec,@noisePolices);
    [~, ~ , policy_hit_rate_theta_noise(i,:)] = runPerfectGuessExp([-0.000001, 0, -1 , 1 ],@noiseCAHMM);
    
    % untrained random theta, trained tandom theta, untrained random tr, 
    hitrate(i,1:2) = [policy_hit_rate_theta_noise(i,1),policy_hit_rate_theta_noise(i,2)]; %N(0,\theta), CA(N(0,\theta))
    hitrate(i,3:4) = [policy_hit_rate_tr_noise(i,6),policy_hit_rate_tr_noise(i,1)]; %N(1,\delta), CA(N(1,\delta))
    hitrate(i,5:6) = [policy_hit_rate_policy_noise(i,6),policy_hit_rate_policy_noise(i,1)]; %N(1,\pi), CA(N(1,\pi))
    hitrate(i,7:8) = [policy_hit_rate_theta_noise(i,3), policy_hit_rate_theta_noise(i,4)]; %N(1,\theta), CA(N(1,\theta))
    
%     hitrate(i,1:2) = runPerfectGuessExpHitRate(0,@noiseCAHMM);
%     hitrate(i,3:4) = runPerfectGuessExpHitRate(1,@noiseTr);
%     hitrate(i,5:6) = runPerfectGuessExpHitRate(1,@noisePolices);
%     hitrate(i,7:8) = runPerfectGuessExpHitRate(1,@noiseCAHMM);
    
end

plot_gramm(hitrate, transJS,policiesJS,noisevec);

end


function hitrate = runPerfectGuessExpHitRate(noiseVec, noisefun)

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

max_iterations = 200;
tolerance = 1e-5;

hitrate = [];


for noise=noiseVec
    
    theta_guess = noisefun(theta_gt, noise);
    
    guess_policies = neurolate_tabular_policies( {theta_guess.eH ,theta_guess.eT});
    
    estimated_policies_seq = scahmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
            theta_guess.trR, theta_guess.trNR, guess_policies);
       
    policy_hit_rate_untrained = mean(MatchingPoliciesHitrate(estimated_policies_seq, testseq, theta_guess, theta_gt));
    

    [theta_hat.trR, theta_hat.trNR, policies_hat, ~] = ...
    scahmmtrain(trainseq.emissions, trainseq.envtype , trainseq.rewards ,theta_guess.trR ,theta_guess.trNR ,...
    guess_policies, 'VERBOSE',false, 'maxiterations', max_iterations, 'TOLERANCE', tolerance);

    table_policies = tabulate_neural_policies(policies_hat, 2);

    theta_hat.eH = table_policies{1};
    theta_hat.eT = table_policies{2};

    estimated_policies_seq = scahmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
            theta_hat.trR, theta_hat.trNR, policies_hat);
       
    policy_hit_rate = mean(MatchingPoliciesHitrate(estimated_policies_seq, testseq, theta_hat, theta_gt));
    
    hitrate = [hitrate, policy_hit_rate_untrained, policy_hit_rate];
end
end


function [transJS, policiesJS, policy_hit_rate] = runPerfectGuessExp(noiseVec, noisefun)

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

max_iterations = 200;
tolerance = 1e-4;

transJS = [];
policiesJS = [];
policy_hit_rate=[];

for noise=noiseVec
    
    
%     if noise<0
%         random_theta =  noiseCAHMM(theta_gt, 1);
%         [trans_JS_noise,policies_JS_noise] = paramatersJS(theta_gt,random_theta);
%         transJS = [transJS, trans_JS_noise];
%         policiesJS = [policiesJS, policies_JS_noise];
%         
%         policies_random = neurolate_tabular_policies({random_theta.eH, random_theta.eT});
%         estimated_policies_seq = scahmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
%             random_theta.trR, random_theta.trNR, policies_random);
%         
%         table_policies = tabulate_neural_policies(policies_hat, 2);
% 
%         theta_hat.eH = table_policies{1};
%         theta_hat.eT = table_policies{2};
%     
%         policy_hit_rate = [policy_hit_rate, mean(MatchingPoliciesHitrate(estimated_policies_seq, testseq, theta_hat, theta_gt))];
%             
%         continue;
%     end
     
    
    if noise>=0
        theta_guess = noisefun(theta_gt, noise);
        
        guess_policies = neurolate_tabular_policies( {theta_guess.eH ,theta_guess.eT});
        
        [theta_hat.trR, theta_hat.trNR, policies_hat, logliks] = ...
        scahmmtrain(trainseq.emissions, num2cell(onehot(trainseq.envtype',1:2),2)' , trainseq.rewards ,theta_guess.trR ,theta_guess.trNR ,...
        guess_policies, 'VERBOSE',false, 'maxiterations', max_iterations, 'TOLERANCE', tolerance);

    else 
        theta_guess = noisefun(theta_gt, abs(noise));
        theta_hat = theta_guess;
        
        policies_hat = neurolate_tabular_policies({theta_hat.eH,theta_hat.eT});
    end
    

    table_policies = tabulate_neural_policies(policies_hat, 2);

    theta_hat.eH = table_policies{1};
    theta_hat.eT = table_policies{2};
 
    [trans_JS_noise,policies_JS_noise] = paramatersJS(theta_gt,theta_hat);
    transJS = [transJS, trans_JS_noise];
    policiesJS = [policiesJS, policies_JS_noise];
    
    estimated_policies_seq = scahmmviterbi(testseq.emissions, testseq.envtype, testseq.rewards,...
            theta_hat.trR, theta_hat.trNR, policies_hat);
       
    policy_hit_rate = [policy_hit_rate, mean(MatchingPoliciesHitrate(estimated_policies_seq, testseq, theta_hat, theta_gt))];
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

function random_theta = getRandomTheta()
random_theta.trR = getrandomdistribution(4,4) ;
random_theta.trNR = getrandomdistribution(4,4);
random_theta.eH = getrandomdistribution(4,2) ;
random_theta.eT = getrandomdistribution(4,2) ;
end

        
function plot_gramm(hitrate, transJS, policiesJS, noisevec)

reps = size(policiesJS,1);
clear g;

%fig 1
types = repmat({'Initial', 'Trained', 'Initial',...
    'Trained','Initial', 'Trained', 'Initial', 'Trained'},reps,1);

% x = repmat({'$N(\theta,0)$','$N(\theta,0)$', '$N(\delta,1)$','$N(\delta,1)$',...
%     '$N(\Pi,1)$','$N(\Pi,1)$', '$N(\theta,1)$','$N(\theta,1)$'},reps,1);

% a - N(\theta,0)
% b - N(\delta,1)
% c - N(\Pi,1)
% d - N(\theta,1)

x = repmat({'(a)','(a)', '(b)','(b)', '(c)','(c)', '(d)','(d)'},reps,1);

g(1,3)=gramm('x', x(:) ,'y',hitrate(:),'color',types(:));
g(1,3).stat_boxplot('dodge',0.7, 'width',0.7)
g(1,3).set_names('x',' ','y','$hitrate$','color','');
g(1,3).axe_property('YLim',[0 1]);

g(1,3).set_layout_options('legend_pos',[0.79 0.1 0.1 0.3])

%g(1,3).set_color_options('map','d3_10');

% figs 2 and 3

categories = repmat({'Trained','Trained',...
    'Trained', 'Trained','Trained',...
    'Initial'},reps,1);

noise_volume = repmat(noisevec,reps,1);
noise_volume(noise_volume<0)=1;

g(1,2)=gramm('x', noise_volume(:) ,'y',transJS(:),'color',categories(:));
g(1,1)=gramm('x', noise_volume(:) ,'y',policiesJS(:), 'color', categories(:));

g(1,2).stat_boxplot('dodge',1, 'width',1.0);
g(1,1).stat_boxplot('dodge',1, 'width',1.0);

g(1,2).set_names('x','$\nu$','y','$JS(\delta,\hat{\delta})$','color','');
g(1,1).set_names('x','$\nu$','y','$JS(\Pi,\hat{\Pi})$','color','');
g.set_text_options('font','Helvetica',...
    'base_size',18,...
    'label_scaling',1.2,...
    'legend_scaling',1,...
    'legend_title_scaling',1,...
    'facet_scaling',1,...
    'title_scaling',1, ...
    'Interpreter', 'latex');
g(1,2).axe_property('YLim',[0 0.3]);
g(1,1).axe_property('YLim',[0 0.3])
g.set_order_options('color',-1);

%g.coord_flip();
g(1,2).set_color_options('map','d3_10');
g(1,1).set_color_options('map','d3_10');
g(1,1).no_legend();
g(1,2).set_layout_options('legend_pos',[0.48 0.1 0.1 0.3])
%g(1,1).axe_property('XTickLabel',[0.25,0.5,1]);
%g(1,2).axe_property('XTickLabel',{'','0.5', '1'});
figure();

g.draw();
set(gca,'TickLabelInterpreter','latex')
g.export('file_name','noising-guess','file_type','png')

end

