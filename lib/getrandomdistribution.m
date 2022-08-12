function D = getrandomdistribution(rows,columns)
%GETRANDONDISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here
rng('shuffle')
D = randg(ones(rows, columns));
D = D./repmat(sum(D,2),1,columns);

end

