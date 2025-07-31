% clc;
% clear all;
% close all;
function seq = gold_gen(Mpn)
%*************** 参数设置 ***************%
% x1(n)序列的初始值，为固定值
x1_init = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

% x2()序列的初始值c_init，由循环前缀CP的类型、小区ID、时隙号和OFDM符号的序号等参数共同决定
% 为了简化，此处使用一个任意的序列
x2_init = [0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];

% Mpn为最后序列c()的长度，自定义
% Mpn = 12 * 10;

% Nc为固定值，ts 36.211中有定义
Nc = 1600;

% 用0初始化x1(n)序列和x2(n)序列，序列长度为Nc + Mpn + 31
x1 = zeros(1,Nc + Mpn + 31);
x2 = zeros(1,Nc + Mpn + 31);

% 用0初始化c(n)序列
seq = zeros(1,Mpn);

% 初始化x1(n)序列和x2(n)序列
x1(1:31) = x1_init;
x2(1:31) = x2_init;

% 生成m序列 : x1()
for n = 1 : (Mpn+Nc)
   x1(n+31) = mod(x1(n+3) + x1(n),2);
end

% 生成m序列 : x2()
for n = 1 : (Mpn+Nc)
   x2(n+31) = mod(x2(n+3) + x2(n+2) + x2(n+1) + x2(n),2);
end

% 用2个m序列生成Gold序列 : c()
for n = 1 : Mpn
    seq(n)= mod(x1(n+Nc) + x2(n+Nc),2);
end 

end