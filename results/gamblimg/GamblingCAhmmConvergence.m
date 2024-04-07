function GamblingCAhmmConvergence()

tr_res = [];
e_res = [];
steps = 15; 
from = 0;
to = 500;

repetition = 25; 

for i=1:repetition
    [trdistances,edistances] = getdistanceforsequence(from,to,steps);
    tr_res=[tr_res;trdistances];
    e_res = [e_res; edistances];
end


save('GamblingCAhmmConvergence_res', 'tr_res', 'e_res')
plot_results(tr_res, e_res, from,to,steps)
end

function plot_results(tr_res, e_res, from, to, steps)
figure;
delta_color = [139,10,80]./255;
pi_color = [69,139,116]./255;
h_delta = shadedErrorBar(linspace(from, to, steps),tr_res,{@mean,@sem}, {'-','Color',delta_color},3);
xlabel('$\ell^{train}$', 'Interpreter','latex')
js_delta='$JS(\delta||\hat{\delta})$';
ylabel('$JS(\delta||\hat{\delta})$','Interpreter','latex')
hold on;
yyaxis right
h_pi = shadedErrorBar(linspace(from, to, steps),e_res,{@mean,@sem},{'-','Color',pi_color},3);
js_pi = '$JS(\Pi||\hat{\Pi})$';
ylabel(js_pi, 'Interpreter','latex')
set(gca,'fontsize',16)
box off;
legend([h_delta.mainLine,h_pi.mainLine],js_delta,js_pi, 'Interpreter','latex')
%title('Parameters Estimation accuracy')

ax = gca;
ax.YAxis(1).Color = delta_color;
ax.YAxis(2).Color = pi_color;
end


function [Trdistance,Edistance] = getdistanceforsequence(from, to, steps)
% checks the correlation between the sequence length and the error in the
% estimated matrices. The longer the sequence the more correct should be the trained model.

%theta_gt = get_neural_random_gt_parameters(2);

theta_gt= get_neural_gt_parameters(); %constant policies.

guess = get_neural_random_gt_parameters();
guess.policies = RandomNetworks(2);

interaction = generate_gambling_trajectory(theta_gt.trR, theta_gt.trNR, theta_gt.policies, to);

max_iter = 50;
Trdistance = [];
Edistance = [];
lengths = linspace(from,to, steps);

for seq_length=floor(lengths)
    
    seq_data.states = interaction.states(1:seq_length,:);
    seq_data.actions = interaction.actions(1:seq_length);
    seq_data.rewards = interaction.wins(1:seq_length);
    
    %CAHMM
    Ddistanceiter = run_cahmm_train(seq_data, theta_gt, guess, max_iter);
    
    
    Trdistance = [Trdistance, Ddistanceiter(1)];
    Edistance = [Edistance, Ddistanceiter(2)];
end
end

function tot_error = run_cahmm_train(seq_data, theta_gt, guess, max_iterations)


[theta_hat.trR, theta_hat.trNR, theta_hat.policies] = ...
    scahmmtrain(seq_data.actions, num2cell(seq_data.states, 2)', seq_data.rewards ,guess.trR ,guess.trNR ,...
    guess.policies, 'VERBOSE', true, 'maxiterations', max_iterations);

theta_hat.policies = NetworksToNeuralPolicies(theta_hat.policies);

theta_gt.policies = NetworksToNeuralPolicies(MimicGamblingPolicy(theta_gt.policies, 1000));
[trans_JS,policies_JS] = paramatersJS(theta_gt,theta_hat);


tot_error = [trans_JS, policies_JS];

end





function tot_error = run_hmm_train(seq_data, guess_trans,  guess_emit, max_iterations)

[realTR, realTRnr, realEhomo, realEhetro] = get_random_gt_parameters(4);

[est_trans, est_emits] = ...
    hmmtrain(seq_data.emissions, guess_trans, guess_emit, 'VERBOSE',false, 'maxiterations', max_iterations);


[policies_match,diff_policies] = MatchandComparePolicies([{realEhomo},{realEhetro}],[{est_emits}, {est_emits}]);

order = policies_match(:,1);
est_trans = est_trans(order,:);

diff_trans = mean([sum(JSDiv(est_trans ,realTR)),sum(JSDiv(est_trans ,realTRnr))]);
tot_error = [diff_trans, mean(diff_policies)];

end
