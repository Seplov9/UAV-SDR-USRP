clc;clear;
addpath("input_baseband")
%% input
% baseband signal
load x_100M.mat
x_DAC=x(:);
% RF input
PA_in=load('1.6b100M_input.txt');
PA_in=PA_in(6401:end-6400);
%% RF output
addpath('PA_wo_dpd')
PA_out1=load("PA_wo_dpd/100M_6_7/d-1.txt");% 10G/55.5dBm
% lowpass filter 
K2  = 260;
Wn2 = [0.28,0.36];%[0.28,0.36] 关于0.32对称，带宽0.2
w2 = fir1(K2,Wn2);
PA_out_down=filter(w2,1,PA_out1);
%% 同步
[aa_corr,aa_point]=xcorr(PA_in,PA_out_down);
[a,b]=max(abs(aa_corr));
b_lag=aa_point(b);
PA_out=PA_out_down((-b_lag+1):(-b_lag+length(PA_in)+0)); % b+1或者b+2  +9
%% down conversion
% carrier 参数设置
len=length(PA_out);
f_carrier =1.6e9; % 载波频率
sampling_rate=10e9;%% 采样率
% 生成时间轴
t = 0:1/sampling_rate:(len-1)/sampling_rate; t=t.';
% 载波
carrier= exp(-1j*2*pi*f_carrier*t); 
PA_down=2*PA_out.*(carrier);
%% 滤波，得到基带的两路I/Q 信号
% lowpass filter
K2  = 260;
Wn2 = 0.25;
w2 = fir1(K2,Wn2,'low');
w2=w2./Wn2;
PA_ADC_upsampling=filter_lowpass_groupdelay(w2,PA_down);
%% 降采样
PA_baseband=downsample(PA_ADC_upsampling,20);

