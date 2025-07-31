function seq = zc_gen(r,len,shn)
%ZC_GEN 生成ZC序列
%   seq = zc_gen(R,LEN,SHN) 产生以质数r为种子，长度为len，向左循环移动shn的ZC
%   序列seq。seq是幅度为1的复数数组。

%   See also ZC_NOR_FFT

%   Copyright 1996-2016 Bridge Qisoo

% --- Initial checks
narginchk(3,3);
seq = zeros(len,1);
remainder = rem(len, 2);
if remainder == 0
	for m = 0:1:(len-1)
		s0index = shn + m;
		if s0index >= len
			s0index = s0index - len;
        end
		seq(m+1) = exp(-1i*pi*r*s0index*s0index/len);
	end
else
	for m = 0:1:(len-1)
		s0index = shn + m;
		if s0index >= len
			s0index = s0index - len;
        end
		seq(m+1) = exp(-1i*pi*r*s0index*(s0index+1)/len);
	end
end