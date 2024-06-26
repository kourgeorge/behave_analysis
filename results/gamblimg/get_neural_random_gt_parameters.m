function theta_gt = get_neural_random_gt_parameters()

eps_r = 0.5 + 0.5*rand(1,1);
eps_nr =  0.5*rand(1,1);

theta_gt.trR  = [eps_r 1-eps_r; eps_r 1-eps_r];
theta_gt.trNR = [eps_nr 1*eps_nr; eps_r 1-eps_r];

policies = [];
for i=1:2
    a = randn;
    b = randn;
    policies = [policies; {LinearPolicy(a,b)}];
end
theta_gt.policies = policies;
end