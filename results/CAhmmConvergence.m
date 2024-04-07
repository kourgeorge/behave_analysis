function CAhmmConvergence()

tr_res = [];
e_res = [];
steps = 20;
from = 0;
to = 1500;

repetition = 25;

for i=1:repetition
    [trdistances,edistances] = getdistanceforsequence(from,to,steps);
    tr_res=[tr_res;trdistances];
    e_res = [e_res; edistances];
end
save('CAhmmConvergence_res', 'tr_res', 'e_res')
plot_results(tr_res, e_res,from,to,steps)

end

function plot_results(tr_res, e_res ,from,to,steps)
figure;
delta_color = [139,10,80]./255;
pi_color = [69,139,116]./255;
h_delta = shadedErrorBar(linspace(from,to,steps),tr_res,{@mean,@sem}, {'-','Color',delta_color},3);
xlabel('$\ell^{train}$', 'Interpreter','latex')
js_delta='$JS(\delta||\hat{\delta})$';
ylabel('$JS(\delta||\hat{\delta})$','Interpreter','latex')
hold on;
yyaxis right
h_pi = shadedErrorBar(linspace(from,to,steps),e_res,{@mean,@sem},{'-','Color',pi_color},3);
js_pi = '$JS(\Pi||\hat{\Pi})$';
ylabel(js_pi, 'Interpreter','latex')
set(gca,'fontsize',14)
box off;
legend([h_delta.mainLine,h_pi.mainLine],js_delta,js_pi, 'Interpreter','latex')
%title('Parameters Estimation accuracy')

ax = gca;
ax.YAxis(1).Color = delta_color;
ax.YAxis(2).Color = pi_color;
end

function [trR, trNR, eH, eT] = get_gt_parameters(N)

eps_r = 0.5 + 0.5*rand(1,1);
eps_nr =  0.5*rand(1,1);

eH = getrandomdistribution(N,2);
eT = getrandomdistribution(N,2);
trR  = ((1-eps_r)/(N-1))*(ones(N)-eye(N)) + eye(N)*eps_r;
trNR = ((1-eps_nr)/(N-1))*(ones(N)-eye(N)) + eye(N)*eps_nr;
end

function [Trdistance,Edistance] = getdistanceforsequence(from, to, steps)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[theta_gt.trR, theta_gt.trNR, theta_gt.eH, theta_gt.eT] = get_gt_parameters(4);
env_type_frac = 0.5;
[envtype, emissions, ~, rewards] = ...
    mazehmmgenerate(to, theta_gt.trR, theta_gt.trNR, ...
    theta_gt.eH, theta_gt.eT ,env_type_frac, [1 0; 0 1]);


guess.trR = NoiseProbabilityMatrix(1, theta_gt.trR);
guess.trNR = NoiseProbabilityMatrix(1, theta_gt.trNR);
guess.eH = NoiseProbabilityMatrix(1, theta_gt.eH);
guess.eT = NoiseProbabilityMatrix(1, theta_gt.eT);


max_iter = 200;
Trdistance = [];
Edistance = [];
lengths = linspace(from,to, steps);

for seq_length=floor(lengths)
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    %CAHMM
    Ddistanceiter = run_cahmm_train(seq_data, theta_gt, guess, max_iter);
    
    % HMM
    %Ddistanceiter = run_maze_train(seq_data, tr_guess, e_guess, max_iter);
    
    Trdistance = [Trdistance, Ddistanceiter(1)];
    Edistance = [Edistance, Ddistanceiter(2)];
end
end

function tot_error = run_cahmm_train(seq_data, theta_gt, guess, max_iterations)

% [theta_hat.trR, theta_hat.trNR, theta_hat.eH, theta_hat.eT] = ...
%     mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
%     guess_emit_homo, guess_emit_hetro, 'VERBOSE',false, 'maxiterations', max_iterations);

guess_policies = neurolate_tabular_policies({guess.eH , guess.eT});
tic
[theta_hat.trR, theta_hat.trNR, models_hat] = ...
    scahmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess.trR  ,guess.trNR ,...
    guess_policies, 'VERBOSE', true, 'maxiterations', max_iterations);
toc
table_policies = tabulate_neural_policies(models_hat, 2);
theta_hat.eH = table_policies{1};
theta_hat.eT = table_policies{2};
[trans_JS,policies_JS] = paramatersJS(theta_gt,theta_hat);


tot_error = [trans_JS, policies_JS];

end

function tot_error = run_hmm_train(seq_data, guess_trans,  guess_emit, max_iterations)

[realTR, realTRnr, realEhomo, realEhetro] = get_gt_parameters(4);

[est_trans, est_emits] = ...
    hmmtrain(seq_data.emissions, guess_trans, guess_emit, 'VERBOSE',false, 'maxiterations', max_iterations);


[policies_match,diff_policies] = MatchandComparePolicies([{realEhomo},{realEhetro}],[{est_emits}, {est_emits}]);

order = policies_match(:,1);
est_trans = est_trans(order,:);

diff_trans = mean([sum(JSDiv(est_trans ,realTR)),sum(JSDiv(est_trans ,realTRnr))]);
tot_error = [diff_trans, mean(diff_policies)];

end
