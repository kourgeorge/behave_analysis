function RunextractEvents

% % for r=1:length(rat)
% for r=1:length(dates)


%   d
dates{1}='004_21012016';
dates{2}='004_22012016';
dates{3}='004_24012016(1)';
dates{4}='004_24012016(2)';
dates{5}='004_25012016(1)';
dates{6}='004_25012016(2)';
dates{7}='004_26012016';
dates{8}='004_27012016(1)';
dates{9}='004_27012016(2)';
dates{10}='004_28012016';
dates{11}='004_29012016';
dates{12}='004_31012016(2)';

%dates{13}='005_01042016';
dates{13}='005_03042016';
dates{14}='005_05042016';
dates{15}='005_07042016(2)';
dates{16}='005_12042016';
dates{17}='005_14042016';

dates{18}='006_03042016';
dates{19}='006_04042016';
dates{20}='006_05042016';
dates{21}='006_06042016';
dates{22}='006_07042016';
dates{23}='006_08042016';
dates{24}='006_11042016';
dates{25}='006_12042016';
dates{26}='006_13042016';
dates{27}='006_13042016(2)';
dates{28}='006_13042016(4)';
dates{29}='006_14042016';
dates{30}='006_20042016(2)';
dates{31}='006_21042016';
dates{32}='006_26042016';
dates{33}='006_27042016';

%dates{34}='007_31032016';
%dates{35}='007_01042016';
dates{34}='007_03042016';
dates{35}='007_04042016';
dates{36}='007_05042016';
dates{37}='007_06042016';
dates{38}='007_07042016';
dates{39}='007_08042016';
dates{40}='007_10042016';
dates{41}='007_12042016';
dates{42}='007_13042016(1)';
dates{43}='007_14042016(1)';
dates{44}='007_20042016';
dates{45}='007_20042016(2)';
dates{46}='007_21042016';
dates{47}='007_25042016';
dates{48}='007_26042016';
dates{49}='007_29042016';

dates{50}='016_28072016';
dates{51}='016_29072016';
dates{52}='027_12052017';
dates{53}='027_13052017';
dates{54}='027_14052017';
dates{55}='027_15052017';

dates{56}='027_17052017';
dates{57}='027_18052017';
dates{58}='027_19052017';
dates{59}='027_22052017';
dates{60}='027_23052017';
dates{61}='027_24052017';
dates{62}='027_25052017';
dates{63}='027_26052017';
dates{64}='027_27052017';
dates{65}='030_12052017';
dates{66}='030_13052017';
dates{67}='030_14052017';
dates{68}='030_15052017';

dates{69}='030_17052017';
dates{70}='030_18052017';
dates{71}='030_19052017';
dates{72}='030_22052017';
dates{73}='030_24052017';
dates{74}='030_25052017';
dates{75}='030_27052017';
dates{76}='031_12052017';

dates{77}='031_17052017';
dates{78}='031_18052017';
dates{79}='031_19052017';
dates{80}='031_22052017';
dates{81}='031_23052017';

for d=1:length(dates)
    datestr=fullfile('E:\experiment\RecordingData',dates{d});
    disp(['processing day ',num2str(d),' of ',num2str(length(dates))]);
    %,' on rat ',num2str(r),' of ',num2str(length(rat)
    extractEvents(datestr);
    
end
end
%EventDate{61}='027_16052017'; dates{74}='030_16052017'; dates{82}='031_16052017';





