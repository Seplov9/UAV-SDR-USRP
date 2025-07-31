clear;
close all;

% Tx data参数情况
N_sc=64;      %系统子载波数
N_fft=N_sc;   % FFT长度
N_cp=16;      % CP长度
data_station=[9:16,21:28,37:44,49:56];    %数据位置
null_station=[1:8,17:20,29:36,45:48,57:64];

% 读取CSV文件
file_path = 'usrp240329/';
data_name = ['3.1G-4M-70m/S/data5.xlsx'];
data_path = [file_path, data_name];
data = readmatrix(data_path);

data_name2 = 'G-64.txt';
data_path2 = [file_path, data_name2];
data2 = readmatrix(data_path2);

data_name3 = 'G-64.mat';
data_path3 = [file_path, data_name3];
Inital_fft = load(data_path3).data;

symbol_num = length(data2)/(N_sc+N_cp);

% tag标签
y=1;

%% 基础数据处理，做降采样率等的初步判断
% 根据原始数据的长度判断截断多少
% 并不准，因为存在oversample和downsample
Inital_data_len = size(data2, 1);
Rec_data_len = size(data, 1);
trunc_len = Inital_data_len * 2;
if trunc_len > size(data, 1)
    trunc_len = size(data, 1);
end

Inital_data_I = data2(:, 1);
Inital_data_Q = data2(:, 2);
Inital_data = Inital_data_I + 1j*Inital_data_Q;
rx0_data_I = data(:,2);  
rx0_data_Q = data(:,4);
rx1_data_I = data(:,6);  
rx1_data_Q = data(:,8);

pos=1;
framenum=0;
CFR_LS = [];
CIR_LS = [];
Inital_fft = Inital_fft + 1e-8+1e-8i;

while pos<=(Rec_data_len-trunc_len-10)

trunc_data0_I = rx0_data_I(pos:pos+trunc_len-1);
trunc_data0_Q = rx0_data_Q(pos:pos+trunc_len-1);
trunc_data0 = trunc_data0_I + 1j*trunc_data0_Q;


trunc_data1_I = rx1_data_I(pos:pos+trunc_len-1);
trunc_data1_Q = rx1_data_Q(pos:pos+trunc_len-1);
trunc_data1 = trunc_data1_I + 1j*trunc_data1_Q;

% 计算channel0

% 通过相关性求信号开始的位置（低SNR时的优选）
% 初始化最大相关值和位置
max_corr = -Inf;
max_pos = 0;

% 滑动窗口大小
win_size = Inital_data_len;

corr_val = zeros(1, trunc_len-Inital_data_len+1);
corr_abs = zeros(1, trunc_len-Inital_data_len+1);
% 滑动窗口互相关
for i = 1:trunc_len-Inital_data_len+1
    corr_val(i) = sum(Inital_data .* conj(trunc_data0(i:i+win_size-1)));
    corr_abs(i) = abs(corr_val(i));
end

% 最大相关位置即为信号开始位置
start_pos = find(corr_abs == max(corr_abs));
data_frame0 = trunc_data0(start_pos:start_pos+Inital_data_len-1);

% 串并转换
    Rx_data1=reshape(data_frame0,N_fft+N_cp,[]);

% 去掉保护间隔、循环前缀
    Rx_data2=Rx_data1(N_cp+1:end,:);
    CP_data=Rx_data1(1:N_cp,:);

% FFT
    fft_data=fft(Rx_data2);

    CFR = fft_data ./ Inital_fft;
    CIR = ifft(CFR);

    CFR_LS = [CFR_LS,CFR];
    CIR_LS = [CIR_LS,CIR];


% 计算channel1

% 通过相关性求信号开始的位置（低SNR时的优选）
% 初始化最大相关值和位置
max_corr = -Inf;
max_pos = 0;

% 滑动窗口大小
win_size = Inital_data_len;

corr_val = zeros(1, trunc_len-Inital_data_len+1);
corr_abs = zeros(1, trunc_len-Inital_data_len+1);
% 滑动窗口互相关
for i = 1:trunc_len-Inital_data_len+1
    corr_val(i) = sum(Inital_data .* conj(trunc_data1(i:i+win_size-1)));
    corr_abs(i) = abs(corr_val(i));
end

% 最大相关位置即为信号开始位置
start_pos = find(corr_abs == max(corr_abs));
data_frame1 = trunc_data1(start_pos:start_pos+Inital_data_len-1);

% 串并转换
    Rx_data1=reshape(data_frame1,N_fft+N_cp,[]);

% 去掉保护间隔、循环前缀
    Rx_data2=Rx_data1(N_cp+1:end,:);
    CP_data=Rx_data1(1:N_cp,:);

% FFT
    fft_data=fft(Rx_data2);

    CFR = fft_data ./ Inital_fft;
    CIR = ifft(CFR);

    CFR_LS = [CFR_LS,CFR];
    CIR_LS = [CIR_LS,CIR];

pos = pos+start_pos+Inital_data_len-10;
framenum=framenum+1;

end

%%
% CFR_real = real(CFR_LS);
% CFR_imag = imag(CFR_LS);
% CFR_pow = abs(CFR_LS);
% 
% idx = 9;
% figure;
% plot(CFR_real(idx, 1:1000))
% hold on
% plot(CFR_imag(idx, 1:1000))
% 
% figure;
% plot(CFR_pow(idx, 1:1000))

%%
X=CFR_LS(data_station(1:end),:);

a = size(X,2);
Y=zeros(1,a);
Y(:)=y;

save(fullfile('usrp240329/3.1G-4M-70m/z-cfr', 'CFR10.mat'),'X','Y');
