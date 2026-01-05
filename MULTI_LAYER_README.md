# 多层地标点检测系统

## 概述

这是一个4层过滤系统，用于从K线数据中提取可靠的高点和低点（地标点），有效过滤趋势中的小波动。

## 4层过滤架构

### 第一层：ZigZag 检测
- **功能**：使用低阈值（5-15%）捕获所有潜在反转点
- **参数**：`layer1_threshold` - ZigZag阈值（默认8%）
- **目的**：尽可能多地捕获候选点，避免遗漏重要地标

### 第二层：统计确认
- **功能**：过滤不显著的点
- **参数**：
  - `layer2_min_frequency` - 最小出现次数（默认2）
  - `layer2_min_deviation_pct` - 最小价格偏离（默认15%）
- **规则**：
  - 价格偏离必须足够大（>15%）
  - 孤立的弱信号被过滤（出现次数<2）
  - 高频或高偏离的信号被保留

### 第三层：趋势过滤
- **功能**：在强趋势中只保留主要反转点
- **参数**：
  - `layer3_min_trend_strength` - 最小趋势强度（默认0.5）
  - `layer3_trend_reversal_threshold` - 趋势反转阈值（默认20%）
- **规则**：
  - 检测局部趋势方向和强度
  - 强下跌趋势：过滤掉偏离趋势线<25%的高点
  - 强上涨趋势：过滤掉偏离趋势线<25%的低点

### 第四层：时间间隔过滤
- **功能**：防止相近的点被重复检测
- **参数**：
  - `layer4_min_weeks_same_type` - 同类型最小间隔（默认10周）
  - `layer4_min_weeks_alternating` - 交替类型最小间隔（默认6周）
- **规则**：
  - 同类型地标（高点-高点或低点-低点）至少间隔10周
  - 交替类型（高点-低点）至少间隔6周
  - 间隔太近时，保留置信度更高的点

## 使用方法

### 方法1：作为Python模块

```python
from core.multi_layer_detector import MultiLayerDetector, LayerFilterParams
import pandas as pd

# 加载价格数据
prices = pd.Series([/* 您的价格数据 */])

# 使用默认参数
detector = MultiLayerDetector()
landmarks, stats = detector.detect(prices, debug=True)

# 自定义参数
params = LayerFilterParams(
    layer1_threshold=0.10,  # 10% ZigZag阈值
    layer2_min_frequency=3,    # 最小频率3次
    layer4_min_weeks_same_type=12  # 同类型12周间隔
)
detector = MultiLayerDetector(params)
landmarks, stats = detector.detect(prices, debug=True)

# 访问结果
for lm in landmarks:
    print(f"{lm['type'].upper()} at week {lm['index']}: "
          f"price={lm['price']:.2f}, layer={lm['layer']}, conf={lm['confidence']:.2f}")
```

### 方法2：快速函数

```python
from core.multi_layer_detector import detect_with_params
import pandas as pd

prices = pd.Series([/* 您的价格数据 */])

landmarks, stats = detect_with_params(
    prices,
    layer1_threshold=0.08,
    layer2_min_freq=2,
    layer2_min_dev=0.15,
    layer3_trend_str=0.5,
    layer4_same_type=10,
    layer4_alt=6,
    debug=True
)
```

### 方法3：交互式Web界面

```bash
cd /Users/apple/study/predict
python -m web.interactive_visualizer
```

然后在浏览器中打开：`http://localhost:5000/interactive`

## 参数调整建议

### 稳健策略（过滤更严格）
- `layer1_threshold`: 10-12%（更高的初始阈值）
- `layer2_min_frequency`: 3-4（要求更高的频率）
- `layer2_min_deviation_pct`: 20-25%（要求更大的偏离）
- `layer3_min_trend_strength`: 0.6-0.7（更强的趋势过滤）
- `layer4_min_weeks_same_type`: 12-16（更长的间隔）

### 灵敏策略（捕获更多点）
- `layer1_threshold`: 5-7%（更低的初始阈值）
- `layer2_min_frequency`: 1-2（允许单次出现）
- `layer2_min_deviation_pct`: 10-12%（允许较小的偏离）
- `layer3_min_trend_strength`: 0.3-0.4（更宽松的趋势过滤）
- `layer4_min_weeks_same_type`: 8-10（更短的间隔）

## 输出格式

### landmarks（地标点列表）
每个地标点包含：
- `index`: 在价格序列中的位置（周数）
- `price`: 价格
- `type`: 'high' 或 'low'
- `layer`: 通过哪一层确认（1-4）
- `confidence`: 置信度（0.0-1.0）

### stats（统计信息）
- `layer1_count`: 第一层候选点数量
- `layer2_count`: 第二层通过数量
- `layer3_count`: 第三层通过数量
- `layer4_count`: 最终数量

## 效果示例

使用默认参数的测试结果：
```
Layer 1 (ZigZag): 4 个候选点
Layer 2 (Statistical): 3 个通过
Layer 3 (Trend): 3 个通过
Layer 4 (Interval): 3 个最终地标点

过滤效果: 25% 的候选点被过滤
```

## 集成到现有系统

将 `landmark_detector.py` 中的检测逻辑替换为多层检测：

```python
from core.multi_layer_detector import MultiLayerDetector

def multi_timeframe_analysis(weekly_df, daily_df, threshold=0.20):
    # 使用多层检测器替代原有的zigzag_detector
    detector = MultiLayerDetector()

    weekly_prices = weekly_df['收盘'] if '收盘' in weekly_df.columns else weekly_df['close']

    landmarks, stats = detector.detect(weekly_prices, debug=False)

    # 继续处理...
    return landmarks
```

## 文件说明

- `core/multi_layer_detector.py` - 核心检测算法
- `web/interactive_visualizer.py` - Web服务器
- `web/templates/interactive.html` - 交互式界面
- `test_multi_layer.py` - 测试脚本（可选）
