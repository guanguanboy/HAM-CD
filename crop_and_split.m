%% 参数设置
patchSize = 256;
overlap = 0.5;
stride = floor(patchSize * (1 - overlap));
testRatio = 0.2; % 预留测试集的粗略比例

% 创建保存目录
mkdir('dataset/train/L8'); mkdir('dataset/train/S1'); mkdir('dataset/train/ROI');
mkdir('dataset/test/L8');  mkdir('dataset/test/S1');  mkdir('dataset/test/ROI');

%% 1. 计算滑动窗口起始坐标
[rows, cols, ~] = size(t1_L8_clipped);
row_indices = 1:stride:(rows - patchSize + 1);
col_indices = 1:stride:(cols - patchSize + 1);

[MeshCol, MeshRow] = meshgrid(1:length(col_indices), 1:length(row_indices));
totalPatches = length(row_indices) * length(col_indices);

%% 2. 划分训练集与测试集 (空间隔离法)
% 为了防止训练集和测试集因重叠而产生信息泄露
% 我们在坐标网格上进行抽样，并排除测试集周围的 8 个邻居
rng(42); % 固定随机种子
allIdx = 1:totalPatches;
testIndices = [];
availableIdx = true(size(MeshRow)); % 可用于抽样的网格标记

numTest = floor(totalPatches * testRatio);

% 迭代抽取测试集并剔除邻域
tempAvailableIdx = find(availableIdx);
for i = 1:numTest
    if isempty(tempAvailableIdx), break; end
    
    % 随机选一个点作为测试集
    sel = tempAvailableIdx(randi(length(tempAvailableIdx)));
    [r, c] = ind2sub(size(availableIdx), sel);
    testIndices = [testIndices; sel];
    
    % 排除该点及其周围 3x3 区域
    r_range = max(1, r-1):min(size(availableIdx,1), r+1);
    c_range = max(1, c-1):min(size(availableIdx,2), c+1);
    availableIdx(r_range, c_range) = false;
    tempAvailableIdx = find(availableIdx);
end

trainIndices = setdiff(allIdx, testIndices);

%% 3. 执行裁剪与保存
all_sets = {trainIndices, testIndices};
set_names = {'train', 'test'};

for s = 1:2
    currIdx = all_sets{s};
    for k = 1:length(currIdx)
        idx = currIdx(k);
        [r_grid, c_grid] = ind2sub(size(MeshRow), idx);
        
        row_start = row_indices(r_grid);
        col_start = col_indices(c_grid);
        row_end = row_start + patchSize - 1;
        col_end = col_start + patchSize - 1;
        
        % 裁剪数据
        patch_L8 = t1_L8_clipped(row_start:row_end, col_start:col_end, :);
        patch_S1 = logt2_clipped(row_start:row_end, col_start:col_end, :);
        patch_ROI = ROI(row_start:row_end, col_start:col_end);
        
        % --- 数据处理与归一化 (转换为 PNG 兼容格式) ---
        % Landsat 8: 取前三波段或代表性波段(如6,5,4)归一化到0-1，再存为uint8
        % 这里演示将11通道数据通过线性拉伸保存，实际使用中可能需要根据特定通道选择
        img_L8 = uint8(rescale(patch_L8(:,:,[6,5,4])) * 255); % 示例：合成假彩色
        img_S1 = uint8(rescale(patch_S1) * 255);             % Sentinel-1 已经是0-1
        img_ROI = uint8(patch_ROI * 255);                    % 二值图转0,255
        
        % 保存文件
        fileName = sprintf('patch_%d_%d.png', row_start, col_start);
        imwrite(img_L8, fullfile('dataset', set_names{s}, 'L8', fileName));
        imwrite(img_S1, fullfile('dataset', set_names{s}, 'S1', fileName));
        imwrite(img_ROI, fullfile('dataset', set_names{s}, 'ROI', fileName));
    end
end

fprintf('处理完成！训练集: %d, 测试集: %d\n', length(trainIndices), length(testIndices));