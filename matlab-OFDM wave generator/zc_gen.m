function seq = zc_gen(r,len,shn)
%ZC_GEN ����ZC����
%   seq = zc_gen(R,LEN,SHN) ����������rΪ���ӣ�����Ϊlen������ѭ���ƶ�shn��ZC
%   ����seq��seq�Ƿ���Ϊ1�ĸ������顣

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