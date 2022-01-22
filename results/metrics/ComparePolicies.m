function dist = ComparePolicies(Policy1, Policy2)
num_states = size(Policy1,1);
sum_divs = 0;
for state=1:num_states
    sum_divs=sum_divs+JSDiv(Policy1(state,:), Policy2(state,:));
end
dist = sum_divs/num_states;

end