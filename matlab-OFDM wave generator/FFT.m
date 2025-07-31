% 生成一个包含两个正弦波的信号
fs = 1000;  % 采样率为1000Hz
t = 0:1/fs:1-1/fs;  % 1秒钟的信号
f1 = 5;  % 第一个正弦波频率为5Hz
f2 = 50;  % 第二个正弦波频率为50Hz
signal = sin(2 * pi * f1 * t) + 0.5 * sin(2 * pi * f2 * t);

% 计算FFT
fft_result = fft(signal);
freq = linspace(0, fs, length(signal));

% 绘制原始信号和频谱
figure;
subplot(2, 1, 1);
plot(t, signal);
title('Original Signal');

subplot(2, 1, 2);
plot(freq, abs(fft_result));
title('Frequency Spectrum');

% 使用IFFT将频谱转换回原始信号
ifft_result = ifft(fft_result);

% 绘制IFFT后的信号
figure;
plot(t, ifft_result);
title('Signal Reconstructed by IFFT');
