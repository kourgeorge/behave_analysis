function x = randprobmatrix( rows, columns )
%RANDPROBMATRIX Returns a probability matrix, in which each row sums to 1.
%   Detailed explanation goes here

x = rand(rows, columns);
totx = sum(x,2);
x = x./repmat(totx,1,columns);

end

