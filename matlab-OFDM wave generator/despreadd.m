function [data] = despread(spread_data, code)
    % 解扩展函数
    % spread_data: 输入的扩展数据序列
    % code: 扩展码序列
    % data: 解扩展后的输出数据序列

    % 检查输入参数数量
    if nargin < 2
        error('缺少输入参数');
    end

    % 获取扩展数据和扩展码的尺寸
    [hsd, vsd] = size(spread_data);
    [hc, vc] = size(code);

    % 检查扩展数据和扩展码的尺寸是否匹配
    if hsd ~= hc
        error('扩展数据和扩展码的行数不匹配');
    end

    % 初始化输出数据数组
    data = zeros(hsd, vsd / vc);

    % 进行解扩展操作
    for ii = 1:hsd
        % 将扩展数据重塑为与扩展码相同的大小
        temp_spread = reshape(spread_data(ii, :), vc, []);

        % 解扩展操作：将重塑后的扩展数据与扩展码相乘，并求和
        data(ii, :) = sum(temp_spread .* repmat(code(ii, :).', 1, size(temp_spread, 2)), 1);
    end
end
