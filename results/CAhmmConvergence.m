function CAhmmConvergence()

tr_res = [];
e_res = [];
steps = 20;
from = 50;
to = 2000;

repetition = 45;

for i=1:repetition
    [trdistances,edistances] = getdistanceforsequence(from,to,steps);
    tr_res=[tr_res;trdistances];
    e_res = [e_res; edistances];
end

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

function [trR, trNR, eH, eT] = get_real_parameters(eps)
eH = [1-eps eps;
    eps 1-eps;
    1-eps eps;
    eps 1-eps];

eT = [1-eps eps; %o1l2 o2l1
    eps 1-eps;
    eps 1-eps;
    1-eps eps];


trR = [0.7 0.1 0.1 0.1;
    0.1 0.7 0.1 0.1;
    0.1 0.1 0.7 0.1;
    0.7 0.1 0.1 0.7];


trNR = [0.1 0.3 0.3 0.3;
    0.3 0.1 0.3 0.3;
    0.3 0.3 0.1 0.3;
    0.3 0.3 0.3 0.1];

end

function [Trdistance,Edistance] = getdistanceforsequence(from, to, steps)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

[realTRr, realTRnr, realEhomo, realEhetro] = get_gt_parameters(4);
env_type_frac = 0.5;
[envtype, emissions, ~, rewards] = ...
    mazehmmgenerate(to, realTRr, realTRnr, ...
    realEhomo, realEhetro ,env_type_frac, [1 0; 0 1]);


tr_guess = getrandomdistribution(4,4);
e_guess = getrandomdistribution(4,2);


max_iter = 200;
Trdistance = [];
Edistance = [];
lengths = linspace(from,to, steps);

for seq_length=floor(lengths)
    
    seq_data.envtype = envtype(1:seq_length);
    seq_data.emissions = emissions(1:seq_length);
    seq_data.rewards = rewards(1:seq_length);
    
    %CAHMM
    Ddistanceiter = run_cahmm_train(seq_data, tr_guess, tr_guess, e_guess, e_guess, max_iter);
    
    % HMM
    %Ddistanceiter = run_maze_train(seq_data, tr_guess, e_guess, max_iter);
    
    Trdistance = [Trdistance, Ddistanceiter(1)];
    Edistance = [Edistance, Ddistanceiter(2)];
end
end

function tot_error = run_cahmm_train(seq_data, guess_trans_reward, guess_trans_noreward, guess_emit_homo, guess_emit_hetro, max_iterations)

[theta_gt.trR, theta_gt.trNR, theta_gt.eH, theta_gt.eT] = get_real_parameters(0.1);

% [theta_hat.trR, theta_hat.trNR, theta_hat.eH, theta_hat.eT] = ...
%     mazehmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
%     guess_emit_homo, guess_emit_hetro, 'VERBOSE',false, 'maxiterations', max_iterations);

guess_policies = neurolate_tabular_policies({guess_emit_homo, guess_emit_hetro});

[theta_hat.trR, theta_hat.trNR, models_hat] = ...
    scahmmtrain(seq_data.emissions, seq_data.envtype , seq_data.rewards ,guess_trans_reward ,guess_trans_noreward ,...
    guess_policies, 'VERBOSE', true, 'maxiterations', max_iterations);

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
