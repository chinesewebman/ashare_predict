# A股形态预测 / A-Share Stock Pattern Predictor

> 基于时间序列分析的A股形态点预测系统 / Time-based sequence analysis for A-share stock landmark prediction

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## English

### Overview

A-Share Stock Pattern Predictor is a web application that predicts WHEN landmark low/high points will occur using time-based sequence analysis. Unlike traditional technical analysis that focuses on price patterns, this system analyzes mathematical patterns in the timing of landmark points.

### Features

- **Multi-layer Landmark Detection**: 4-layer progressive filtering system
  - Layer 1: ZigZag algorithm for initial pivot detection
  - Layer 2: Statistical confirmation (frequency & deviation filters)
  - Layer 3: Trend strength filtering
  - Layer 4: Time interval validation

- **Pattern Recognition**: Identifies arithmetic and geometric patterns in landmark timing

- **Bilingual Interface**: Full support for English and Chinese

- **Real-time Analysis**: FastAPI backend with responsive frontend

- **Interactive Charts**: Chart.js visualization with landmark annotations

### Tech Stack

**Backend:**
- Python 3.10+
- FastAPI
- pandas, numpy
- scikit-learn
- matplotlib

**Frontend:**
- HTML5, JavaScript (Vanilla)
- Bootstrap 5
- Chart.js 4.4.0

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ashare_predict.git
cd ashare_predict
```

2. Create virtual environment:
```bash
conda create -n ashare python=3.10
conda activate ashare
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare data directory:
```bash
mkdir -p data/weekly
# Place your weekly OHLCV data in data/weekly/ directory
# Format: {stock_code}.csv with columns: 日期, 开盘, 最高, 最低, 收盘, 成交量
```

5. Run the application:
```bash
cd web
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

6. Open browser:
```
http://localhost:8000
```

### Usage

1. **Search Stock**: Enter stock code (6 digits), pinyin, or name
2. **View Prediction**: Click "分析" to see pattern prediction
3. **Detailed Analysis**: Click "查看完整分析" for full chart and landmark details
4. **Adjust Parameters**: Use multi-layer filter panel to fine-tune detection

### Project Structure

```
ashare_predict/
├── core/                   # Core analysis modules
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── zigzag.py           # ZigZag algorithm implementation
│   ├── pattern_detector.py # Pattern recognition engine
│   └── multi_layer_detector.py # Multi-layer filtering
├── web/                    # Web application
│   ├── app.py              # FastAPI application
│   ├── templates/          # HTML templates
│   └── static/             # CSS, JS, images
├── data/                   # Data directory
│   └── weekly/             # Weekly stock data
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### API Endpoints

- `GET /` - Homepage
- `GET /analysis/{symbol}` - Stock analysis page
- `GET /api/predict?symbol={code}` - Get prediction
- `GET /api/analyze?symbol={code}&threshold={val}&include_secondary={bool}` - Get full analysis
- `GET /api/stocks/search?q={query}` - Search stocks
- `POST /api/multi-layer-detect` - Run multi-layer detection with custom parameters

### Default Parameters

Optimized parameters based on testing:
- Layer 1 Threshold: 10%
- Layer 2 Min Frequency: 2
- Layer 2 Min Deviation: 20%
- Layer 3 Trend Strength: 0.50
- Layer 4 Same Type Interval: 10 weeks
- Layer 4 Alternating Interval: 4 weeks

### Disclaimer

⚠️ **This is NOT Financial Advice**

This tool is for educational and research purposes only. Past patterns do not guarantee future results. Always do your own research and consult a qualified financial advisor before making investment decisions.

### License

MIT License

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<a name="chinese"></a>
## 中文

### 概述

A股形态预测系统是一个基于时间序列分析的Web应用，用于预测关键低点/高点出现的时间。与关注价格形态的传统技术分析不同，本系统分析形态点时机的数学规律。

### 功能特点

- **多层地标点检测**：4层渐进式过滤系统
  - 第1层：ZigZag算法初步检测转折点
  - 第2层：统计确认（频率和偏离度过滤）
  - 第3层：趋势强度过滤
  - 第4层：时间间隔验证

- **模式识别**：识别地标点时机中的算术和几何规律

- **双语界面**：完整支持中英文

- **实时分析**：FastAPI后端，响应式前端

- **交互式图表**：Chart.js可视化，带地标点标注

### 技术栈

**后端：**
- Python 3.10+
- FastAPI
- pandas, numpy
- scikit-learn
- matplotlib

**前端：**
- HTML5, JavaScript (原生)
- Bootstrap 5
- Chart.js 4.4.0

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/ashare_predict.git
cd ashare_predict
```

2. 创建虚拟环境：
```bash
conda create -n ashare python=3.10
conda activate ashare
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 准备数据目录：
```bash
mkdir -p data/weekly
# 将周线OHLCV数据放入 data/weekly/ 目录
# 格式: {股票代码}.csv，包含列: 日期, 开盘, 最高, 最低, 收盘, 成交量
```

5. 运行应用：
```bash
cd web
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

6. 打开浏览器：
```
http://localhost:8000
```

### 使用方法

1. **搜索股票**：输入股票代码（6位数字）、拼音或名称
2. **查看预测**：点击"分析"按钮查看形态预测
3. **详细分析**：点击"查看完整分析"查看完整图表和地标点详情
4. **调整参数**：使用多层过滤控制面板微调检测参数

### 项目结构

```
ashare_predict/
├── core/                   # 核心分析模块
│   ├── data_loader.py      # 数据加载和预处理
│   ├── zigzag.py           # ZigZag算法实现
│   ├── pattern_detector.py # 模式识别引擎
│   └── multi_layer_detector.py # 多层过滤系统
├── web/                    # Web应用
│   ├── app.py              # FastAPI应用
│   ├── templates/          # HTML模板
│   └── static/             # CSS、JS、图片
├── data/                   # 数据目录
│   └── weekly/             # 周线股票数据
├── requirements.txt        # Python依赖
└── README.md              # 本文件
```

### API接口

- `GET /` - 首页
- `GET /analysis/{symbol}` - 股票分析页面
- `GET /api/predict?symbol={code}` - 获取预测
- `GET /api/analyze?symbol={code}&threshold={val}&include_secondary={bool}` - 获取完整分析
- `GET /api/stocks/search?q={query}` - 搜索股票
- `POST /api/multi-layer-detect` - 使用自定义参数运行多层检测

### 默认参数

基于测试优化的默认参数：
- 第1层阈值：10%
- 第2层最小频率：2
- 第2层最小偏离：20%
- 第3层趋势强度：0.50
- 第4层同类型间隔：10周
- 第4层交替间隔：4周

### 免责声明

⚠️ **这不是投资建议**

本工具仅供教育和研究目的。过去的规律不能保证未来的结果。在做出投资决策前，务必自行研究并咨询合格的专业理财顾问。

### 许可证

MIT License

### 贡献

欢迎贡献！请随时提交 Pull Request。

---

**Developed with ❤️ for A-share market analysis**
