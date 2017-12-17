function plotPsthByStrategy2()

%Example: plotPsthByStrategy2('D:\lab', '004_26012016', 'TT1 - Copy_SS_03.ntt' , '004_26012016-strategy.csv')

baseFolder = 'D:\lab';
EventDate='004_28012016';
SpikeClust='TT8 - Copy_SS_04.ntt';
behaviorFileName = '004_26012016-strategy.csv';
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
events = events.eventsStruct;

events = cleanRedundantEvents(events);

%load the stability intervals
stability = load(fullfile(eventfile, ['stability_', SpikeClust, '.mat']));


%load the behavior file
behave_data = csvread(fullfile(baseFolder,EventDate,behaviorFileName), 1, 0);
strategy_column = 9;
strategies = behave_data(1:end-1, strategy_column); % avoid the last line - it contains noise, the strategy is in column 9

% select an event

        
for q=1:length(stability.stStart)
    
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
        [Psth, Spikes] = CalcPSTHAroundEvent( event, st, 1.0, 1.0, win, 2000);
        
        figure('pos',[10 10 900 600])
        drawPSTHandSpikes(Psth,Spikes,1,1,2000)
        set(0,'DefaultTextInterpreter','none')
        suptitle(['Event: ', eventName,'. Strategy: ',num2str(strategy), ' Cell: ', SpikeClust ,' Day: ', EventDate])
        saveas(gcf,[eventName,num2str(strategy),'.jpg'])        
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function drawPSTHandSpikes(psth,spikes,range1,range2,timecut)
subplot(2,1,1)
bar(linspace(-range1,range2,timecut),psth,'k');
axis tight; box off; grid off;
subplot(2,1,2)
spy(spikes,'k.')
axis normal
xlabel = -timecut/2: 500 :timecut/2;
set(gca,'XTick', xlabel);
end

function [eventInInterval, eventInIntervalRewarded, eventInIntervalNonRewarded] = ...
    getEventSpikeInterval(eventData, rewardedTrials, startPoint, endPoint)
%we assume that the trial number is in the last field.
eventInInterval = getSpikeInterval(eventData, startPoint, endPoint);
eventInIntervalRewarded = getSpikeInterval(eventData(ismember(eventData(:,end),rewardedTrials),:), startPoint, endPoint);
eventInIntervalNonRewarded = getSpikeInterval(eventData(~ismember(eventData(:,end),rewardedTrials),:), startPoint, endPoint);

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

function events = cleanRedundantEvents(events)

%keep the last Ain
trials = events.aBeamEnter(:,end);
indicesLasteventInTrial = getLastInSeq(trials);
events.aBeamEnter = events.aBeamEnter(indicesLasteventInTrial,:);

%keep the first Bin
trials = events.bBeamEnter(:,end);
indicesFirsteventInTrial = getFirstInSeq(trials);
events.bBeamEnter = events.bBeamEnter(indicesFirsteventInTrial,:);


%keep the last bout
trials = events.bBeamExit(:,end);
indicesLasteventInTrial = getLastInSeq(trials);
events.bBeamExit = events.bBeamExit(indicesLasteventInTrial,:);

%keep the first Aout
trials = events.aBeamExit(:,end);
indicesFirsteventInTrial = getFirstInSeq(trials);
events.aBeamExit = events.aBeamExit(indicesFirsteventInTrial,:);

end


function ind = getFirstInSeq(arr)
ind  = find([1;arr(2:end)-arr(1:end-1)]);
end

function ind = getLastInSeq(arr)
ind  = find([arr(2:end)-arr(1:end-1);1]);
end