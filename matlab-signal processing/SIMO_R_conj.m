clear;
close all;

% Tx data参数情况
N_sc=64;      %系统子载波数
N_fft=N_sc;   % FFT长度
N_cp=16;      % CP长度
data_station=[9:16,21:28,37:44,49:56];    %数据位置
null_station=[1:8,17:20,29:36,45:48,57:64];


X_all=[];
Y_all=[];


%% DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD
% 读取CSV文件
file_path = 'usrp240409/';
% data_name = '1.6G/N/data2.xlsx';
% data_path = [file_path, data_name];
% data = readmatrix(data_path);

data_name2 = 'G-64.txt';
data_path2 = [file_path, data_name2];
data2 = readmatrix(data_path2);

symbol_num = length(data2)/(N_sc+N_cp);

% tag标签
y=0;

% 设置包含 .mat 文件的文件夹路径
data_folder = 'usrp240409/-5db/D/';

% 获取文件夹中所有 .xlsx 文件
xlsx_files = dir(fullfile(data_folder, '*.xlsx'));

Inital_data_len = size(data2, 1);
Inital_data_I = data2(:, 1);
Inital_data_Q = data2(:, 2);
Inital_data = Inital_data_I + 1j*Inital_data_Q;

totalRounds = length(xlsx_files);
Receive_data0 = [];
Receive_data1 = [];

for round = 1:totalRounds

data_path = [data_folder, xlsx_files(round).name];
data = readmatrix(data_path);

%% 基础数据处理，做降采样率等的初步判断
% 根据原始数据的长度判断截断多少
% 并不准，因为存在oversample和downsample
Inital_data_len = size(data2, 1);
Rec_data_len = size(data, 1);
trunc_len = Inital_data_len * 2;
if trunc_len > size(data, 1)
    trunc_len = size(data, 1);
end

rx0_data_I = data(:,2);  
rx0_data_Q = data(:,4);
rx1_data_I = data(:,6);  
rx1_data_Q = data(:,8);

pos=1;
framenum=0;
h = waitbar(0, sprintf('轮次 D-%d/%d 开始...', round, totalRounds));

while pos<=(Rec_data_len-trunc_len-10)
% while pos<=10000

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

Receive_data0 = [Receive_data0,data_frame0];


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

Receive_data1 = [Receive_data1,data_frame1];

pos = pos+start_pos+Inital_data_len-10;
framenum=framenum+1;

t=framenum/(Rec_data_len/Inital_data_len);
waitbar(t, h, sprintf('当前进度：轮次 D-%d/%d, 进度 %d%%', round, totalRounds, floor(t*100)));

end

close(h);

end

%%
R_c = [];

for i=1:size(Receive_data0, 2)
    
    R_c_TEMP = Receive_data0(:,i) .* conj(Receive_data1(:,i));

    R_c = [R_c,R_c_TEMP];

end

a = size(R_c,2);
Y=zeros(1,a);
Y(:)=y;

X_all = [X_all,R_c];
Y_all = [Y_all,Y];




%% SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
% 读取CSV文件
file_path = 'usrp240409/';
% data_name = '1.6G/N/data2.xlsx';
% data_path = [file_path, data_name];
% data = readmatrix(data_path);

data_name2 = 'G-64.txt';
data_path2 = [file_path, data_name2];
data2 = readmatrix(data_path2);

symbol_num = length(data2)/(N_sc+N_cp);

% tag标签
y=1;

% 设置包含 .mat 文件的文件夹路径
data_folder = 'usrp240409/-5db/M/';

% 获取文件夹中所有 .xlsx 文件
xlsx_files = dir(fullfile(data_folder, '*.xlsx'));

Inital_data_len = size(data2, 1);
Inital_data_I = data2(:, 1);
Inital_data_Q = data2(:, 2);
Inital_data = Inital_data_I + 1j*Inital_data_Q;

totalRounds = length(xlsx_files);
Receive_data0 = [];
Receive_data1 = [];

for round = 1:totalRounds

data_path = [data_folder, xlsx_files(round).name];
data = readmatrix(data_path);

%% 基础数据处理，做降采样率等的初步判断
% 根据原始数据的长度判断截断多少
% 并不准，因为存在oversample和downsample
Inital_data_len = size(data2, 1);
Rec_data_len = size(data, 1);
trunc_len = Inital_data_len * 2;
if trunc_len > size(data, 1)
    trunc_len = size(data, 1);
end

rx0_data_I = data(:,2);  
rx0_data_Q = data(:,4);
rx1_data_I = data(:,6);  
rx1_data_Q = data(:,8);

pos=1;
framenum=0;
h = waitbar(0, sprintf('轮次 S-%d/%d 开始...', round, totalRounds));

while pos<=(Rec_data_len-trunc_len-10)
% while pos<=10000

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

Receive_data0 = [Receive_data0,data_frame0];


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

Receive_data1 = [Receive_data1,data_frame1];

pos = pos+start_pos+Inital_data_len-10;
framenum=framenum+1;

t=framenum/(Rec_data_len/Inital_data_len);
waitbar(t, h, sprintf('当前进度：轮次 S-%d/%d, 进度 %d%%', round, totalRounds, floor(t*100)));

end

close(h);

end

%%
R_c = [];

for i=1:size(Receive_data0, 2)
    
    R_c_TEMP = Receive_data0(:,i) .* conj(Receive_data1(:,i));

    R_c = [R_c,R_c_TEMP];

end

a = size(R_c,2);
Y=zeros(1,a);
Y(:)=y;

X_all = [X_all,R_c];
Y_all = [Y_all,Y];



%%
X=X_all.';
Y=Y_all;

% 获取Z的大小
[m, n] = size(X);

% 初始化一个m*2n的实数矩阵
X_real = zeros(m, 2*n);

% 遍历Z，将每个复数的实部和虚部放入Z_real
for i = 1:m
    for j = 1:n
        % 计算新矩阵中的列索引
        colIndex = 2*j-1;
        % 提取实部和虚部
        X_real(i, colIndex) = real(X(i,j));
        X_real(i, colIndex + 1) = imag(X(i,j));
    end
end
X=X_real;

save(fullfile('usrp231210/0db/usrp240409/-5db/D-M', 'R_c.mat'),'X','Y');
