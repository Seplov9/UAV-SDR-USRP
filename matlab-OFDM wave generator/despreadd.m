function [data] = despread(spread_data, code)
    % ����չ����
    % spread_data: �������չ��������
    % code: ��չ������
    % data: ����չ��������������

    % ��������������
    if nargin < 2
        error('ȱ���������');
    end

    % ��ȡ��չ���ݺ���չ��ĳߴ�
    [hsd, vsd] = size(spread_data);
    [hc, vc] = size(code);

    % �����չ���ݺ���չ��ĳߴ��Ƿ�ƥ��
    if hsd ~= hc
        error('��չ���ݺ���չ���������ƥ��');
    end

    % ��ʼ�������������
    data = zeros(hsd, vsd / vc);

    % ���н���չ����
    for ii = 1:hsd
        % ����չ��������Ϊ����չ����ͬ�Ĵ�С
        temp_spread = reshape(spread_data(ii, :), vc, []);

        % ����չ�����������ܺ����չ��������չ����ˣ������
        data(ii, :) = sum(temp_spread .* repmat(code(ii, :).', 1, size(temp_spread, 2)), 1);
    end
end
