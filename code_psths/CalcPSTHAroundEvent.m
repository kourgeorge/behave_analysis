function [PsthAroundEvent,Spikes] = CalcPSTHAroundEvent( events, st, range1, range2, win, timecut)
%PSTHplusMaze_byEvent Given the spike data and events timestamps
%return the PSTH aroung the events timestamps.

eventNum = size(events,1);
Spikes=sparse(eventNum, timecut);

for r=1:eventNum
    TE=events(r,2);
    trange=[TE-range1 TE+range2] ;
    inxsp=st>trange(1)&st<=trange(2);
    s=st(inxsp);
    Spikes(r,max(1,floor((s-trange(1))*1000)))=1;
end

if (eventNum)>=5
    pst = sum(full(Spikes))/size(Spikes,1)*1000;
    PsthAroundEvent=filtfilt(win,1,pst);
else
    disp('empty event')
    PsthAroundEvent = zeros(1, timecut);
end
