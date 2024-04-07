function plotPsthByStrategy()

baseFolder = 'D:\lab';
EventDate='004_31012016';
SpikeClust='TT8 - Copy_SS_04.ntt';
eventName = 'aBeamExit';

sigma=20;
win=gauss(sigma);

%load the cell file
spikesFilePath = fullfile(baseFolder, EventDate, SpikeClust);
st=Nlx2MatSpike(spikesFilePath,[1 0 0 0 0],0,1,[]);
st=st/1e6;

%load the event file
eventfile=fullfile(baseFolder,EventDate);
events = load(fullfile(eventfile, 'events.mat'));

%load the stability intervals
stability = load(fullfile(eventfile, ['stability_', SpikeClust, '.mat']));

strategies = events.strategy;
psth_all = [];
subplot_loc = [1,2,5,6];
for q=1:length(stability.stStart)
    
    figure('pos',[10 10 900 600])
    set(0,'DefaultTextInterpreter','none')
    suptitle(['Event: ', eventName,'. Cell: ', SpikeClust, ...
        ' Day: ', EventDate])
    
    for strategy=1:4
        
        trialsWithStrategy = find(strategies==strategy); %get all the trials that are in strategy
        eventsStructForStrategy = keepTrialsWithinStrategyInEventsStruct(events, trialsWithStrategy);
        
        stStart = stability.stStart;
        stEnd = stability.stEnd;
        
        % select the interval in the neural data
        [~,closest2start]=min(abs(st- stStart(q)));
        [~,closest2end]=min(abs(st-stEnd(q)));
        st = st(closest2start:closest2end);
        
        rewardedTrials = unique(eventsStructForStrategy.rewards(:,3));
        
        
        event = eventsStructForStrategy.(eventName);
        [event, eventCorrect, eventInCorrect] = getEventSpikeInterval(event, rewardedTrials, stStart(q), stEnd(q));
        
        event = eventCorrect;
        [Psth, Spikes] = CalcPSTHAroundEvent( event, st, 1.0, 1.0, win, 2000);
        
        psth_all = [psth_all;Psth];
        
        range1 = 1;
        range2 = 1;
        timecut = 2000;
        subplot(5,2,subplot_loc(strategy))
        drawPSTH(Psth,range1,range2,timecut)
        title(['S', num2str(strategy), '    #events=', num2str(size(event,1)), '     #nz=', num2str(nnz(Spikes))])
        subplot(5,2,subplot_loc(strategy)+2)
        drawSpikes(Spikes, timecut)  

    end
    [Psth, Spikes] = CalcPSTHAroundEvent( events.(eventName), st, 1.0, 1.0, win, 2000);
    
    subplot(5,2,9)
    hold on;
    plot(linspace(-1,1,2000), psth_all', 'linewidth', 2)
    bar(linspace(-1,1,2000),Psth,'k');
    axis tight;
    ylim([0 15])
    legend('S1', 'S2', 'S3', 'S4')
     title(['#events=', num2str(size(events.(eventName),1)), '     #nz=', num2str(nnz(Spikes))])
    hold off;
    subplot(5,2,10)
    drawSpikes(Spikes, timecut)
    %saveas(gcf,[EventDate,SpikeClust,eventName,num2str(strategy),'.jpg'])
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function drawPSTH(psth,range1,range2,timecut)
bar(linspace(-range1,range2,timecut),psth,'k');
axis tight; box off; grid off;
ylim([0 15])
end

function drawSpikes(spikes, timecut)
spy(spikes,'k.')
xlabel('')
xticks([0 1000 2000])
xticklabels({'-1', '0', '1'})
axis normal
end

function [eventInInterval, eventInIntervalRewarded, eventInIntervalNonRewarded] = ...
    getEventSpikeInterval(eventData, rewardedTrials, startPoint, endPoint)
%we assume that the trial number is in the last field.
eventInInterval = getSpikeInterval(eventData, startPoint, endPoint);
eventInIntervalRewarded = eventInInterval(ismember(eventInInterval(:,end),rewardedTrials),:);
eventInIntervalNonRewarded = eventInInterval(~ismember(eventInInterval(:,end),rewardedTrials),:);

end

function interval = getSpikeInterval(eventData, startPoint, endPoint)
% Get the lines in the eventData that are between start and end point
% interval
eventDataTimes = eventData(:,2);
[~,closest2start]=min(abs(eventDataTimes-startPoint));
[~,closest2end]=min(abs(eventDataTimes-endPoint));
interval=eventData(closest2start:closest2end,:);

end

function gaussFilter = gauss(sigma)
width = round((6*sigma - 1)/2);
support = (-width:width);
gaussFilter = exp( -(support).^2 ./ (2*sigma^2) );
gaussFilter = gaussFilter/ sum(gaussFilter);
end

function eventsWithStrategy = getEventsWithinStrategy(eventData, trialsWithStrategy)
% Assuming that in the eventData struct the trial number in the last column
eventsWithStrategy =  eventData(ismember(eventData(:,end), trialsWithStrategy),:);
end

function eventStruct = keepTrialsWithinStrategyInEventsStruct(eventStruct, trialsWithStrategy)
% get the events that are were performed in trials that are in trialsWithStrategy

fields = fieldnames(eventStruct);
for fn=fields'
    eventData = eventStruct.(fn{1});
    if (size(eventData,2)>3) %If num columns is larger than 3 then it is aBeam, bBeam and NP data
        eventStruct.(fn{1}) = getEventsWithinStrategy(eventData, trialsWithStrategy);
    end
end
end
