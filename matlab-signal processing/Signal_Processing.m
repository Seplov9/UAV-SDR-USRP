clear;
close all;

% Tx data参数情况
N_sc=64;      %系统子载波数
N_fft=N_sc;   % FFT长度
N_cp=16;      % CP长度
data_station=[9:16,21:28,37:44,49:56];    %数据位置
null_station=[1:8,17:20,29:36,45:48,57:64];

% 读取文件
file_path = 'usrp240308/';
data_name = '150m/M/data1.xlsx';
data_path = [file_path, data_name];
data = readmatrix(data_path);

data_name2 = 'G-64.txt';  % 发射数据txt
data_path2 = [file_path, data_name2];
data2 = readmatrix(data_path2);

data_name3 = 'G-64.mat';  % 发射数据mat
data_path3 = [file_path, data_name3];
Inital_fft = load(data_path3).data;

symbol_num = length(data2)/(N_sc+N_cp);

%% 基础数据处理，做降采样率等的初步判断
% 根据原始数据的长度判断截断多少
Inital_data_len = size(data2, 1);
trunc_len = Inital_data_len * 2;
if trunc_len > size(data, 1)
    trunc_len = size(data, 1);
end

Inital_data_I = data2(:, 1);
Inital_data_Q = data2(:, 2);
Inital_data = Inital_data_I + 1j*Inital_data_Q;

% 获取I/Q数据
rx_data_I = data(:,2);  
rx_data_Q = data(:,4);
rx1_data_I = data(:,6);  
rx1_data_Q = data(:,8);

trunc_data_I = rx_data_I(1:trunc_len);
trunc_data_Q = rx_data_Q(1:trunc_len);
trunc_data = trunc_data_I + 1j*trunc_data_Q;

%% 通过相关性求信号开始的位置
% 初始化最大相关值和位置
max_corr = -Inf;
max_pos = 0;

% 滑动窗口大小
win_size = Inital_data_len;

corr_val = zeros(1, trunc_len-Inital_data_len+1);
corr_abs = zeros(1, trunc_len-Inital_data_len+1);
% 滑动窗口互相关
for i = 1:trunc_len-Inital_data_len+1
    corr_val(i) = sum(Inital_data .* conj(trunc_data(i:i+win_size-1)));
    corr_abs(i) = abs(corr_val(i));
end

% 最大相关位置即为信号开始位置
start_pos = find(corr_abs == max(corr_abs));
data_frame = trunc_data(start_pos:start_pos+Inital_data_len-1);
% 
%%
% trunc_data1_I = rx1_data_I(1:trunc_len);
% trunc_data1_Q = rx1_data_Q(1:trunc_len);
% trunc_data1 = trunc_data1_I + 1j*trunc_data1_Q;
% 
% max_corr = -Inf;
% max_pos = 0;
% 
% % 滑动窗口大小
% win_size = Inital_data_len;
% 
% corr_val = zeros(1, trunc_len-Inital_data_len+1);
% corr_abs = zeros(1, trunc_len-Inital_data_len+1);
% % 滑动窗口互相关
% for i = 1:trunc_len-Inital_data_len+1
%     corr_val(i) = sum(Inital_data .* conj(trunc_data1(i:i+win_size-1)));
%     corr_abs(i) = abs(corr_val(i));
% end
% 
% % 最大相关位置即为信号开始位置
% start_pos = find(corr_abs == max(corr_abs));
% data_frame1 = trunc_data1(start_pos:start_pos+Inital_data_len-1);

% 
% % 串并转换
%     Rx_data1=reshape(data_frame,N_fft+N_cp,[]);
% 
% % 去掉保护间隔、循环前缀
%     Rx_data2=Rx_data1(N_cp+1:end,:);
%     CP_data=Rx_data1(1:N_cp,:);
% 
% % FFT
%     fft_data=fft(Rx_data2);
% 

%%
load whaleTrill
% Fs=400000;
p = audioplayer(data_frame,Fs,16);
play(p)

whaleTrill=data_frame;

t = (0:length(whaleTrill)-1)/Fs;
figure
ax1 = subplot(3,1,1);
plot(t,whaleTrill)
ax2 = subplot(3,1,2);
pspectrum(whaleTrill,Fs,'spectrogram','OverlapPercent',0, ...
    'Leakage',1,'MinThreshold',-60)
colorbar(ax2,'off')
ax3 = subplot(3,1,3);
pspectrum(whaleTrill,Fs,'spectrogram','OverlapPercent',0, ...
    'Leakage',1,'MinThreshold',-60,'TimeResolution', 10e-3)
colorbar(ax3,'off')
linkaxes([ax1,ax2,ax3],'x')

xlim([0.3 0.55])

%%
% load batsignal
% Fs = 1/DT;
% figure
% % pspectrum(data_frame,Fs,'spectrogram')
% pspectrum(data_frame,Fs,'spectrogram','FrequencyResolution',3e3, ...
%     'OverlapPercent',99,'MinTHreshold',-60)

%%
% load splat
% % y = data_frame;
% p = audioplayer(y,Fs,16);
% play(p)
% pspectrum(y,Fs,'spectrogram')
% 
% rng('default')
% t = (0:length(y)-1)/Fs;
% yNoise = y + 0.1*randn(size(y));
% yChirp = yNoise(t<0.35);
% pspectrum(yChirp,Fs,'spectrogram','MinThreshold',-70)
% 
% fsst(yChirp,Fs,'yaxis')
% 
% [sst,f] = fsst(yChirp,Fs); 
% [fridge, iridge] = tfridge(sst,f,10);
% helperPlotRidge(yChirp,Fs,fridge);
% 
% yrec = ifsst(sst,kaiser(256,10),iridge,'NumFrequencyBins',1);
% pspectrum(yrec,Fs,'spectrogram','MinThreshold',-70)
% 
% p = audioplayer([yChirp;zeros(size(yChirp));yrec],Fs,16);
% play(p);

%%
% Fs = 1e8;
% bw = 60e6;
% t = 0:1/Fs:10e-6;
% IComp = chirp(t,-bw/2,t(end), bw/2,'linear',90)+0.15*randn(size(t));
% QComp = chirp(t,-bw/2,t(end), bw/2,'linear',0) +0.15*randn(size(t));
% IQData = IComp + 1i*QComp;
% % IQData= data_frame;
% 
% segmentLength = 128;
% pspectrum(IQData,Fs,'spectrogram','TimeResolution',1.27e-6,'OverlapPercent',90)

% %%
% Fs = 1e3;                    
% t = 0:1/Fs:10;               
% fo = 10;                     
% f1 = 400;                    
% y = chirp(t,fo,10,f1,'logarithmic');
% % y = data_frame;
% pspectrum(y,Fs,'spectrogram','FrequencyResolution',1, ...
%     'OverlapPercent',90,'Leakage',0.85,'FrequencyLimits',[1 Fs/2])
% ax = gca;
% ax.YScale = 'log';

%%
% Fs = 10e3;
% t = 0:1/Fs:2;
% x1 = vco(sawtooth(2*pi*t,0.5),[0.1 0.4]*Fs,Fs);
% pspectrum(data_frame,Fs,'spectrogram','Leakage',0.8)
% 
% view(-45,65)
% colormap bone

%%
% fs = 1000;
% t = (0:1/fs:500)';
% x = chirp(t,180,t(end),220) + 0.15*randn(size(t));
% idx = floor(length(x)/6);
% x(1:idx) = x(1:idx) + 0.05*cos(2*pi*t(1:idx)*210);
% %x=data_frame;
% pspectrum(x,fs,'FrequencyLimits',[100 290])
% figure
% colormap parula
% pspectrum(x,fs,'persistence','FrequencyLimits',[100 290],'TimeResolution',1)

%%
% fs = 3000;
% t = 0:1/fs:1-1/fs;
% x1 = chirp(t,300,t(end),1300,'quadratic')+randn(size(t))/100;
% x2 = exp(2j*pi*100*cos(2*pi*2*t))+randn(size(t))/100;
% % x1 = data_frame;
% % x2 = data_frame1;
% 
% nwin = 256;
% xspectrogram(x1,x2,kaiser(nwin,30),nwin-1,[],fs,'centered','yaxis')
% xspectrogram(x1,x2,kaiser(nwin,30),nwin-1,[],fs, ...
%     'power','MinThreshold',-40,'yaxis')
% title('Cross-Spectrogram of Quadratic Chirp and Complex Chirp')