# VLM Attention Analysis Summary

## Task 2: 识别负责左上角物体检测的Attention Head和Layer

## 实验设置
- **模型**: LLaVA-1.5-7B
- **测试图片对**:
  - 1.jpg: 左上角有白色三角形，中下方有红色方块
  - 2.jpg: 只有中间的红色方块，左上角无物体
- **问题**: "Is there anything in the top left corner?"
- **模型回答**:
  - 1.jpg: "Yes, there is a triangle in the top left corner of the image."
  - 2.jpg: "No, there is nothing in the top left corner of the image."

## 主要发现

### 1. Per-Head Attention Analysis（最核心发现）

**Top 5 最敏感的Attention Heads**（按左上角attention差异排序）：

| Rank | Layer | Head | Diff | With Triangle | Without Triangle |
|------|-------|------|------|---------------|------------------|
| 1 | 24 | 17 | 0.4832 | 0.8042 | 0.3210 |
| 2 | 23 | 20 | 0.4280 | 0.7754 | 0.3474 |
| 3 | 19 | 31 | 0.4055 | 0.5908 | 0.1853 |
| 4 | 17 | 13 | 0.3677 | 0.6733 | 0.3057 |
| 5 | 17 | 30 | 0.3496 | 0.5947 | 0.2451 |

**关键发现**:
- **Layer 24, Head 17** 是最敏感的head，当有三角形时对左上角的attention为80.42%，无三角形时仅31.10%，差异达48.32%
- **Layer 23, Head 20** 同样表现出很强的空间位置敏感性
- 最敏感的heads主要集中在**后期层（Layer 17-26）**，说明高层特征对空间位置判断更重要
- 早期层（Layer < 15）的heads差异相对较小

### 2. Spatial Attention Distribution（空间分布）

**四象限Attention分布对比**:

| 象限 | With Triangle | Without Triangle | Difference |
|------|---------------|------------------|------------|
| Top-Left | 0.2752 | 0.1929 | **+0.0822** |
| Top-Right | 0.1579 | 0.2135 | -0.0556 |
| Bottom-Left | 0.3966 | 0.3338 | +0.0628 |
| Bottom-Right | 0.1703 | 0.2598 | -0.0895 |

**关键发现**:
- 有三角形时，模型对**左上角的attention增加8.22%**
- 同时，模型对**右下角的attention减少8.95%**（可能因为红色方块在中下位置）
- Bottom-Left象限在两种情况下都获得最多attention（约34-40%），这可能与问题"top left corner"中的"left"一词有关

### 3. Vision Encoder Layer-wise Analysis（CLIP各层分析）

**Vision Encoder对左上角的关注度（按层）**:

关键观察:
- **Layer 0-11**: 差异为正（有三角形时关注度更高）
  - 最大差异在 **Layer 8-9** (差异约7%)
- **Layer 12-21**: 差异为负或接近0
  - Layer 18差异最负（-6.93%）
- **Layer 22-23**: 又回到正差异

**关键发现**:
- Vision encoder的**中间层（Layer 4-11）**对左上角三角形的存在最敏感
- **Layer 8 (差异=0.0713)** 和 **Layer 9 (差异=0.0693)** 是最关键的layers
- 后期层（13-21）反而表现出"反向"趋势，可能是在做更高级的语义整合

### 4. Attention Divergence（分布差异量化）

**Overall Spatial Attention**:
- KL Divergence: 1.7054
- JS Divergence: 0.4653

**Vision Encoder Layer-wise Divergence (Top 5)**:

| Layer | KL Divergence | JS Divergence |
|-------|---------------|---------------|
| 22 | 1.3115 | 0.4271 |
| 19 | 0.6189 | 0.3433 |
| 17 | 0.4830 | 0.3054 |
| 18 | 0.4583 | 0.3003 |
| 16 | 0.4433 | 0.2913 |

**关键发现**:
- **Layer 22** 显示出最大的KL散度（1.31），说明该层在两种情况下的attention模式差异最大
- **Layer 17-19** 也显示出显著差异
- 早期层（0-10）的divergence较小，说明低级视觉特征处理相似

## 核心结论

### 负责左上角物体检测的关键组件：

1. **LLM Attention Heads**:
   - **主要**: Layer 24 Head 17, Layer 23 Head 20
   - **辅助**: Layer 19 Head 31, Layer 17 Head 13/30
   - **特点**: 集中在模型的后期层（Layer 17+），负责高级空间推理

2. **Vision Encoder Layers**:
   - **主要**: Layer 8-9（中间层，提取中级视觉特征）
   - **辅助**: Layer 4-7（早期到中期）
   - **特点**: 中间层对物体位置最敏感

3. **信息流路径**:
   ```
   Vision Encoder Layer 8-9 (检测到左上角物体)
           ↓
   Vision features传递到LLM
           ↓
   LLM Layer 17-24 的特定heads (空间位置推理)
           ↓
   生成正确的Yes/No回答
   ```

## 建议的进一步分析

1. **Head组合分析**: 研究Layer 24 Head 17和Layer 23 Head 20是否协同工作
2. **Token-level追踪**: 分析生成"Yes"/"No"/"triangle"等关键token时的attention模式
3. **Attention Rollout**: 计算从输出token到输入image patches的完整attention路径
4. **更多样本验证**: 在更多左上角物体检测任务上验证发现的heads是否一致

## 生成的文件

所有分析结果保存在 `attention_data/` 目录:
- `attention_with_triangle.pt` - 原始attention数据（有三角形）
- `attention_without_triangle.pt` - 原始attention数据（无三角形）
- `per_head_analysis.pt` - Per-head分析结果
- `spatial_vision_analysis.pt` - 空间分布和vision encoder分析结果
- `top_different_heads.png` - Top 20差异最大的heads可视化
- `spatial_distribution.png` - 空间attention分布对比
- `vision_encoder_layers.png` - Vision encoder各层趋势图
- `divergence_analysis.png` - KL/JS散度分析图
