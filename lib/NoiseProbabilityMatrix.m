function [ moisedmat ] = NoiseProbabilityMatrix( noiseVal, mat )
%NOISEPROBABILITYMATRIX Summary of this function goes here
%   Detailed explanation goes here

[m,n] = size(mat);

temp = (1-noiseVal).*mat + noiseVal.* rand(m,n);
denom = repmat(sum(mat, 2), [1, n]);
moisedmat = temp./denom;

end

