clear
clc
close all
% %% 参数设置
% N_sc=16;      %系统子载波数
% N_fft=N_sc;   % FFT长度
% N_cp=4;       % CP长度
% data_station=[3,4,6,7,10,11,13,14];    %数据位置
% pilot_station=[5,12];                  %导频位置

%% 参数设置2
N_sc=64;      %系统子载波数
N_fft=N_sc;   % FFT长度
N_cp=16;       % CP长度
data_station=[9:16,21:28,37:44,49:56];    %数据位置
pilot_station=[17:20,45:48];              %导频位置


%% 基带数据数据产生
% 生成ZC序列
Nd=6;
N_frm=10;
outputsize = N_sc*Nd*N_frm;
zc_seed = 23; % 选一个质数作为生成ZC的种子
shift_num = 10; % 循环位移次数
outZcSequence = zc_gen(zc_seed, outputsize, shift_num);
modu_data=outZcSequence;

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
Tx_cd=[ifft_data(N_fft-N_cp+1:end,:);ifft_data];%把ifft的末尾N_cp个数补充到最前面
 
%% 并串转换
Tx_data=reshape(Tx_cd,[],1);%由于传输需要

% 获取当前时间
currentTime = datestr(now,'yyyymmddTHHMMSS'); 

%%
[data_corr, lags] = xcorr(Tx_data, Tx_data);
data_corr_abs = abs(data_corr);
figure;
plot(lags, data_corr_abs)
subtitle('ZC corr peak')

max_corr = -Inf;
max_pos = 0;
Tx_data2=[Tx_data;Tx_data];
%xcorr(modu_data, modu_data)

% 滑动窗口大小
win_size = length(Tx_data);

corr_val = zeros(1, win_size);
corr_abs = zeros(1, win_size);
% 滑动窗口互相关
for i = 1:win_size+1
    %corr_val(i) = xcorr(modu_data, modu_data2(i:i+win_size-1));
    corr_val(i) = sum(Tx_data .* conj(Tx_data2(i:i+win_size-1)));
    corr_abs(i) = abs(corr_val(i));
end

figure;
plot(corr_abs)
%%

% 将OFDM Wave保存为文本文件
% fid = fopen(strcat(currentTime,'_ZC OFDM Wave.txt'),'w');
% for k = 1:length(Tx_data)
%     fprintf(fid, '%f,%f\n', real(Tx_cd(k)), imag(Tx_cd(k)));
% end
% fclose(fid);

% nfft = 16;  % FFT length
% cplen = 4; % Cyclic prefix length
% nSym  = 120; % Number of symbols per RE
% 
% nullIdx  = [1 2 8 9 15 16]';
% pilotIdx = [5 12]';
% 
% Nd=6;
% N_frm=10;
% outputsize = nfft*Nd*N_frm;
% zc_seed = 3; % 选一个质数作为生成ZC的种子
% shift_num = 10; % 循环位移次数
% OutZcSequence = zc_gen(zc_seed, outputsize, shift_num);
% numDataCarrs = nfft-length(nullIdx)-length(pilotIdx);
% Data_seq=reshape(OutZcSequence,numDataCarrs,nSym);
% 
% P_f=3+3*1i;  %Pilot frequency
% pilots=ones(2,nSym)*P_f;%将导频放入矩阵
% 
% y2 = ofdmmod(Data_seq,nfft,cplen,nullIdx,pilotIdx,pilots);
% 
% an1=angle(Tx_data);
% an2=angle(y2);
% an=an1-an2;
% disp(an);

