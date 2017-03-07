function D = getrandomdistribution(rows,columns)
%GETRANDONDISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here

D = abs(randn(rows, columns));
D = D./repmat(sum(D,2),1,columns);

end

