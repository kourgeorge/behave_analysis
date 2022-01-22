function matches_distances = MatchAndCompareProbabilities(Pmat1,Pmat2)
%MATCHANDCOMPAREPROBABILITIES Summary of this function goes here
%   Detailed explanation goes here


N = size(Pmat1,1);
for i=1:N
    for j=1:N
        dist(i, j) = KLDiv(Pmat1(i, :),Pmat2(j, :));
    end
end

[match,uR,uC] = matchpairs(dist, 50);
%match = [match;[uR,uC]];
matches_distances = [];
for i=1:N
    matches_distances(i) = KLDiv(Pmat1(match(i,1), :),Pmat2(match(i,2), :));
end

end

