%% File info

% File name:        importOdor.m
% Author:           Boris Boltyansky
% Contact:          boltyansky@gmail.com
% Last modified:    2016-01-09

function varargout = importOdor(varargin)
% IMPORTODOR MATLAB code for importOdor.fig
%      IMPORTODOR, by itself, creates a new IMPORTODOR or raises the existing
%      singleton*.
%
%      H = IMPORTODOR returns the handle to a new IMPORTODOR or the handle to
%      the existing singleton*.
%
%      IMPORTODOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMPORTODOR.M with the given input arguments.
%
%      IMPORTODOR('Property','Value',...) creates a new IMPORTODOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before importOdor_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to importOdor_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help importOdor

% Last Modified by GUIDE v2.5 09-Jan-2016 10:23:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @importOdor_OpeningFcn, ...
                   'gui_OutputFcn',  @importOdor_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before importOdor is made visible.
function importOdor_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to importOdor (see VARARGIN)

% Choose default command line output for importOdor
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes importOdor wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = importOdor_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

handles.scriptDir = pwd; % Save the directory of the script
guidata(hObject,handles);

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[handles.filename, handles.pathname] = uigetfile('*.*',...
    'Select the file you want to convert from text to csv', 'MultiSelect', 'on');
set(handles.pushbutton5,'Enable','on');
guidata(hObject,handles);
set(handles.edit1,'String',strcat(handles.pathname, handles.filename));
guidata(hObject,handles); 

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles=guidata(hObject);

%% Open file
for file=handles.edit1.String'
filefullpath = file{1};
fileID  = fopen(filefullpath, 'r');
data    = textscan(fileID, '%s', 'delimiter', '');
data    = cellstr(data{:});

%% Regex magic

regIn   = '\n';
regOut  = 'XXXXXXXX';
data    = cellfun(@(x) regexprep(x, regIn, regOut), data, 'un', 0);

regIn   = '\s+';
regOut  = '\,';
data    = cellfun(@(x) regexprep(x, regIn, regOut), data, 'un', 0);

regIn   = '\,XXXXXXXX';
regOut  = '\n';
data    = cellfun(@(x) regexprep(x, regIn, regOut), data, 'un', 0);

%% Prepare for export to CSV

data    = cellfun(@(x) strsplit(x, ','), data, 'un', 0);

%# Pad length to equalize lengths
%# ( based on http://stackoverflow.com/a/3054577
%# and on http://stackoverflow.com/a/6210539 )
maxLen  = max(cellfun(@(x) numel(x), data));
data    = cellfun(@(x) cat(2, x,...
    num2cell(zeros(1, maxLen - length(x)))), data, 'un', 0);
NaNFcn  = @(x) [x nan(1, maxLen - numel(x))];
data    = cellfun(NaNFcn, data, 'un', 0);
data    = vertcat(data{:});

%# Extract data only
data = str2double(data(33:end, 2:end));

%# Extract sub matrices between NaNs
%# ( Based on http://www.mathworks.com/matlabcentral/answers/63325-extract-sub-matrices-of-matrix-between-nan-into-cell-array )
pos     = [true, isnan(data(:, 1)).', true];
ini     = strfind(pos, [true, false]);
fin     = strfind(pos, [false, true]) - 1;
sData   = cell(1, length(ini));

for iC = 1:length(ini)
  sData{iC} = data(ini(iC):fin(iC), :);
end

%# Extract the main data matrix
sData   = sData{3}';
sData   = reshape(sData, 1, []);

%# Transform into 10 columns
%# ( Based on http://stackoverflow.com/a/19466267 and on
%# https://www.mathworks.com/matlabcentral/newsreader/view_thread/156573 )
cols_n  = 10;
pad     = @(x) [x, nan(1, cols_n - mod(numel(x), cols_n))];
sData   = pad(sData)'
sData   = reshape(sData, cols_n, [])';

%# Extract columns 1,2,4,5,6,7,9,10
sData                                       = sData(:,[1:2 4:7 9:10]);
sData(sData == 0)                           = nan;
sData                                       = num2cell(sData);
sData(cellfun(@(x) any(isnan(x)),sData))    = {''}

%% Export to CSV

sData = cell2table(sData,...
    'VariableNames', {'trial' 'r' 'w' 'c' 'np' 'cup' 'ic_1' 'ic_2'});

writetable(sData, strcat(filefullpath,'.csv'),...
    'WriteVariableNames', 1);

handles.close_status = fclose(fileID);

set(handles.pushbutton5,'String','Done!');
set(handles.pushbutton3,'Enable','off');
set(handles.pushbutton5,'Enable','off');

guidata(hObject,handles);

end
% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

close(handles.figure1);
