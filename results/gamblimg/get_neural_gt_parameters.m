function theta_gt= get_neural_gt_parameters()


theta_gt.trR  = [0.9 0.1; 0.7, 0.3];
theta_gt.trNR = [0.3 0.7; 0.1, 0.9];

theta_gt.policies = [{LinearPolicy(-1,1)}; {LinearPolicy(1,0)}];

end

