function [st, stStart, stEnd] = determineStabilityrRegions( spikeClustPath, ITI, clust, EventDate )
%DETERMINESTABILITYRREGIONS  Enabke the user to select regions of stability.
% It returns an array of Interval starts and ends, stStart and stEnd,
% respectively.

st=Nlx2MatSpike(spikeClustPath,[1 0 0 0 0],0,1,[]);

st=st/1e6;
timeVector=ITI(1,2):ITI(size(ITI,1),2);
S1=st(1);
E1=st(end);

filename = strcat(date, clust);
for t=1:length(timeVector)-1
    spikesInT(t)=length(find(st>=timeVector(t)&st<timeVector(t+1)));
end
cusum=cumsum(spikesInT);
h = figure;
plot(timeVector(2:end),cusum);
suptitle(['stability',EventDate,', ',clust] )
saveas(h,'1_1.jpg')
pause;
close(h);
stStart = [];
stEnd = [];
%if exist([eventfile 'S3'],'var')%(thirdStart)
if exist('S3')%(thirdStart)
    stStart=[S1(1,1) S2(1,1) S3(1,1)];
    stEnd=[E1(1,1) E2(1,1) E3(1,1)];
    %elseif exist([eventfile 'S2'],'var')%(secondStart)
elseif exist('S2')%(secondStart)
    stStart=[S1(1) S2(1)];
    stEnd=[E1(1) E2(1) ];
else
    stStart= S1(1,1);
    stEnd= E1(1,1);
end

end

