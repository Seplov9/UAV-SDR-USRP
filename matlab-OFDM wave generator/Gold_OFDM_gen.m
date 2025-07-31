clear
clc
close all
% %% 参数设置 16子载波
% N_sc=16;      %系统子载波数
% N_fft=N_sc;   % FFT长度
% M=4;          %4PSK调制
% N_cp=4;       % CP长度
% data_station=[3,4,6,7,10,11,13,14];    %数据位置
% pilot_station=[5,12];                  %导频位置

%% 参数设置 64子载波
N_sc=64;      %系统子载波数
N_fft=N_sc;   % FFT长度
M=4;          %4PSK调制
N_cp=16;       % CP长度
data_station=[9:16,21:28,37:44,49:56];    %数据位置
pilot_station=[17:20,45:48];              %导频位置

%% 基带数据数据产生
% 生成gold序列
Nd=6;
N_frm=10;
outputsize = N_sc*Nd*N_frm;
outGoldSequence = gold_gen(outputsize);

%% qpsk调制
data_temp1= reshape(outGoldSequence,log2(M),[])'; %以每组2比特进行分组，M=4
data_temp2= bi2de(data_temp1);                    %二进制转化为十进制
modu_data=pskmod(data_temp2,M,pi/M);              % 4PSK调制

data_row=length(data_station);
data_col=ceil(length(modu_data)/data_row);
if data_row*data_col>length(modu_data)
    data2=[modu_data;zeros(data_row*data_col-length(modu_data),1)];  %将数据矩阵补齐
else
    data2=modu_data;
end

%% 插入导频
%P_f=3+3*1i;  %Pilot frequency
P_f=0;
pilot_num=length(pilot_station);  %导频数量
pilot_seq=ones(pilot_num,data_col)*P_f;  %将导频放入矩阵

data=zeros(N_fft,data_col);  %预设整个矩阵
data(pilot_station(1:end),:)=pilot_seq;  %对pilot_seq按行取

%% 串并转换
data_seq=reshape(data2,data_row,data_col);
data(data_station(1:end),:)=data_seq;  %将导频与数据合并

%% IFFT
ifft_data=ifft(data); 

%% 插入保护间隔、循环前缀
Tx_cd=[ifft_data(N_fft-N_cp+1:end,:);ifft_data];  %把ifft的末尾N_cp个数补充到最前面

%% 并串转换
Tx_data=reshape(Tx_cd,[],1);  %由于传输需要

%%
[data_corr, lags] = xcorr(Tx_data, Tx_data);
data_corr_abs = abs(data_corr);
figure;
plot(lags, data_corr_abs)
subtitle('Gold corr peak')

% 获取当前时间
currentTime = datestr(now,'yyyymmddTHHMMSS'); 

% % 将OFDM Wave保存为文本文件
% fid = fopen(strcat(currentTime,'_gold OFDM Wave.txt'),'w');
% for k = 1:length(Tx_data)
%     fprintf(fid, '%f,%f\n', real(Tx_cd(k)), imag(Tx_cd(k)));
% end
% fclose(fid);

% %% DAC1
% x = Tx_data;
% % upsample
% up_sample_rate0 = 5;
% x_up=upsample(x,up_sample_rate0);
% % lowpass filter
% K = 260;
% Wn = 1/up_sample_rate0;%修改值，获得不同情况下的band limit效果
% w = fir1(K,Wn,'low');
% gain =up_sample_rate0; w=gain.*w;
% % DAC output
% x_DAC=filter_lowpass_groupdelay(w,x_up); % 500MHz，2.5GHz 采样率
% %% upsample 生成 10GHz 采样率信号
% % upsample
% up_sample_rate1 = 10;
% x_DAC_up=upsample(x_DAC,up_sample_rate1);
% % lowpass filter
% K1 = 260;
% Wn1 = 1/up_sample_rate1;%修改0.1的值，获得不同情况下的band limit效果
% w1 = fir1(K1,Wn1,'low');
% gain1 =up_sample_rate1; w1=gain1.*w1;
% % input
% x_DAC_oversample=filter_lowpass_groupdelay(w1,x_DAC_up); %10GHz 采样率

%% DAC2
x = Tx_data;
% upsample
up_sample_rate = 50;
x_DAC_oversample=upsample(x,up_sample_rate);

%% Upconversion
% carrier 参数设置
len=length(x_DAC_oversample);
f_carrier =1.6e9; % 载波频率
sampling_rate=5e9;%% 采样率
% 生成时间轴
t = 0:1/sampling_rate:(len-1)/sampling_rate; t=t.';
% 载波
carrier= exp(1j*2*pi*f_carrier*t); 
%% PA IF input 100MHz，10GHz 采样率信号
PA_input_IF=real(x_DAC_oversample.*carrier); %% PA_input
% figure(1)
% pwelch(PA_input_IF,[],[],[],sampling_rate);
%% 补零
PA_input_IF=[zeros(2e4,1);PA_input_IF;zeros(2e4,1)];

% A = PA_input_IF(PA_input_IF ~= 0);
% % 打开文本文件用于写入
% fid = fopen('AWGdata1.txt','w');
% % 更改数字显示格式 (不使用科学计数法)
% format short g;
% % 遍历数组并写入文本文件 
% for i = 1:size(A,1)
% fprintf(fid,'%.8f\n',A(i));
% end
% % 关闭文件
% fclose(fid);