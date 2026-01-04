# 空间关系复杂度对Attention模式的影响分析

## 研究概述

本研究通过对比实验，分析了复杂空间关系（left/right/above/below）与简单位置关系（adjacent）对Vision-Language Model (LLaVA) attention模式的影响，重点关注attention的**entropy（熵）**和**dispersion（分散度）**两个指标。

## 实验设计

### 实验假设
- **复杂关系**（left/right/above/below）：需要精确的空间定位，可能产生更集中、更有序的attention模式
- **简单关系**（adjacent）：只需要判断相邻关系，可能产生更分散、更无序的attention模式

### 实验方法
1. **数据准备**：从`data/spatial_twoshapes/agreement/relational/test/`加载包含复杂空间关系的样本
2. **Caption替换**：将原始caption中的复杂关系词替换为"adjacent"
   - 原始：`"a gray ellipse is to the left of a blue circle ."`
   - 替换后：`"a gray ellipse is adjacent to a blue circle ."`
3. **对比分析**：对同一图像，分别使用原始caption和替换后的caption提取attention，计算并对比metrics

## 分析的Attention Map类型

### 1. Relation Token Cross-Attention
- **定义**：Vision tokens (queries) 对关系词token (key) 的cross-attention
- **时刻**：Forward pass阶段（非generation阶段）
- **含义**：表示视觉patches如何关注空间关系词（如"left"/"right"或"adjacent"）
- **Shape**：`[grid_size, grid_size]` (24×24)

### 2. Entity Token Cross-Attention
- **定义**：Vision tokens (queries) 对实体tokens (keys) 的cross-attention
- **时刻**：Forward pass阶段
- **含义**：表示视觉patches如何关注实体（entity1和entity2）
- **Shape**：`[grid_size, grid_size]` (24×24)

### 3. Vision Self-Attention
- **定义**：Vision token (query) 对vision tokens (keys) 的self-attention
- **时刻**：Forward pass阶段
- **含义**：表示一个vision patch如何关注其他vision patches
- **起始点**：从entity中心位置（entity center seed）出发
- **Shape**：`[grid_size, grid_size]` (24×24)

## 计算的Metrics

### 1. Entropy (熵)
**定义**：Shannon entropy，衡量attention分布的随机性/不确定性

**计算公式**：
```
Entropy = -Σ(p * log(p))
```
其中：
- `p` 是归一化后的attention分布（概率分布）
- 添加小常数（1e-10）避免log(0)

**解释**：
- **高熵**：attention分布更均匀、更随机，信息量更大
- **低熵**：attention分布更集中、更有序，信息量更小

### 2. Dispersion (分散度)
**定义**：衡量attention在空间上的分散程度，类似于attention分布的"空间方差"

**计算步骤详解**：

#### Step 1: 归一化Attention Map
```python
attn = attn_map / (attn_map.sum() + 1e-6)
```
将attention map归一化为概率分布，使得所有值之和为1。

#### Step 2: 计算Attention的质心（Center of Mass）
```python
# 创建坐标网格
ys = torch.arange(H).view(-1, 1)  # [H, 1] 列坐标
xs = torch.arange(W).view(1, -1)    # [1, W] 行坐标

# 计算加权平均坐标（质心）
y_mean = (attn * ys).sum()  # attention在y方向的加权平均
x_mean = (attn * xs).sum()  # attention在x方向的加权平均
```
质心 `(y_mean, x_mean)` 表示attention分布的"重心"位置。

#### Step 3: 计算空间方差（Dispersion）
```python
# 计算每个位置到质心的欧氏距离平方
var = ((ys - y_mean)**2 + (xs - x_mean)**2) * attn

# 对所有位置加权求和
dispersion = var.sum()
```

**完整数学表达**：
```
Dispersion = Σ_{i=0}^{H-1} Σ_{j=0}^{W-1} [((i - ȳ)² + (j - x̄)²) * p_{i,j}]
```
其中：
- `(i, j)` 是网格位置坐标（i是行，j是列）
- `(ȳ, x̄)` 是attention的质心坐标
- `p_{i,j}` 是位置(i,j)的归一化attention值

**物理意义**：
- Dispersion衡量attention分布相对于其质心的"空间扩散程度"
- 类似于概率分布的二阶矩（second moment），但考虑的是二维空间

**直观理解**：
1. **Dispersion = 0**：所有attention完全集中在质心一个点上（理想情况，实际不会发生）
2. **小Dispersion**：attention分布紧密围绕质心，空间上很集中
3. **大Dispersion**：attention分布远离质心，空间上很分散

**示例**：
- 如果attention集中在图像中心的一个小区域（如3×3），dispersion会很小
- 如果attention均匀分布在整个24×24网格上，dispersion会很大（接近理论最大值）
- 如果attention集中在图像边缘，dispersion也会较大（因为距离质心远）

**与Entropy的关系**：
- **Entropy**：衡量attention分布的"信息量"或"随机性"（不考虑空间位置）
- **Dispersion**：衡量attention分布的"空间扩散程度"（考虑空间位置）
- 两者可以互补：一个分布可能entropy高但dispersion低（均匀但集中），或entropy低但dispersion高（集中但位置分散）

## 实验流程

```
原始Caption (left/right/above/below)
    ↓
提取Attention Maps
    ↓
计算Entropy & Dispersion
    ↓
替换为Adjacent Caption
    ↓
提取Attention Maps (相同图像)
    ↓
计算Entropy & Dispersion
    ↓
对比分析 & 可视化
```

## 输出结果

### 1. 数据文件
- `results/attention_analysis_results.csv`：所有样本的详细metrics
- `results/comparison_statistics.csv`：统计对比结果（mean, std, t-test）

### 2. 可视化图表
- `results/comparison_barplots.png`：对比柱状图（原始组 vs 替换组）
- `results/comparison_boxplots.png`：箱线图（展示分布差异）
- `results/entropy_vs_dispersion_scatter.png`：散点图（entropy vs dispersion关系）

## 预期发现

通过对比分析，我们期望发现：

1. **Relation Token Attention**：
   - 复杂关系可能产生更低的entropy（更集中）和更低的dispersion（更聚焦）
   - 简单关系可能产生更高的entropy（更分散）和更高的dispersion（更扩散）

2. **Entity Token Attention**：
   - 复杂关系可能使模型更关注实体的精确位置
   - 简单关系可能使模型对实体位置的关注更分散

3. **Vision Self-Attention**：
   - 复杂关系可能产生更结构化的空间attention模式
   - 简单关系可能产生更均匀的attention模式

## 技术细节

### 模型配置
- **Model**: LLaVA-v1.5-7b
- **Grid Size**: 24×24 (576 vision patches)
- **Attention Extraction**: Forward pass阶段，平均所有layers和heads

### 数据处理
- **数据源**: `data/spatial_twoshapes/agreement/relational/test/`
- **过滤条件**: 只分析agreement=1.0的样本
- **关系类型**: left, right, above, below → adjacent

### 统计方法
- **描述性统计**: Mean, Standard Deviation
- **假设检验**: Independent t-test（比较两组差异的显著性）
- **显著性水平**: p < 0.05 认为有显著差异

## 文件说明

### 主要脚本
- `spatial_relation_analysis.py`: 主分析脚本
  - 批量处理数据
  - 提取attention maps
  - 计算metrics
  - 生成可视化

### 辅助函数
- `vlm_atten_fix.py`: 
  - `compute_entropy()`: 计算entropy
  - `compute_disperson()`: 计算dispersion
  - `extract_prompt_level_cross_attention()`: 提取cross-attention
  - 其他辅助函数

### 数据加载
- `dataloader.py`: 从shard目录加载图像和caption

脚本实现了以下功能：
1. 加载数据
2. 过滤复杂关系样本
3. 对每个样本进行原始和替换caption的分析
4. 计算所有metrics
5. 生成统计报告和可视化

## 注意事项

1. **Token匹配**：由于tokenization可能将单词分割，代码实现了智能匹配算法来找到entity和relation tokens
2. **World Data**：每个shard有独立的`world_model.json`，包含实体的ground truth信息
3. **计算资源**：处理大量样本可能需要较长时间，建议先用`max_samples`参数测试
4. **GPU要求**：需要CUDA支持的GPU来运行LLaVA模型

## 后续分析方向

1. **分层分析**：分析不同transformer layers的attention模式差异
2. **关系类型细分**：分别分析left/right和above/below的差异
3. **实体类型影响**：分析不同形状/颜色的实体对attention的影响
4. **可视化增强**：生成attention map的热力图对比

