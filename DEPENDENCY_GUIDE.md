# 佛山AI项目依赖管理指南

本文档提供佛山AI项目的依赖管理说明，包括依赖配置、安装方法和常见问题解决方案。

## 依赖配置文件

项目使用两种依赖配置文件：

1. **pyproject.toml** - 现代Python项目标准依赖配置文件
2. **requirements.txt** - 传统依赖列表文件

两个文件包含相同的依赖，但格式不同。推荐优先使用`pyproject.toml`进行依赖管理。

## 依赖安装方法

### 方法1：使用pip和pyproject.toml（推荐）

```bash
# 开发模式安装（可编辑模式）
pip install -e .

# 正式安装
pip install .
```

### 方法2：使用requirements.txt

```bash
# 标准安装
pip install -r requirements.txt

# 如需GPU支持
# 1. 先安装CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. 再安装其他依赖
pip install -r requirements.txt --no-deps torch torchvision
```

### 方法3：离线安装

```bash
# 1. 下载依赖包
pip download -r requirements.txt -d ./packages

# 2. 离线安装
pip install --no-index --find-links ./packages -r requirements.txt
```

## 依赖验证

项目提供了依赖验证脚本，用于检查所有必需的依赖是否已正确安装：

```bash
python verify_dependencies.py
```

## 依赖分类

项目依赖分为以下几类：

### 核心数据处理库
- **numpy** - 数值计算基础库，用于数据处理和模型计算
- **pandas** - 数据分析和处理，用于时序数据操作

### 数据库连接
- **influxdb** - InfluxDB时序数据库客户端，用于数据存储和查询

### 配置文件处理
- **PyYAML** - YAML配置文件解析，用于configs.yaml和oa_args.yaml

### 机器学习框架
- **torch** - PyTorch深度学习框架，用于MLP、GRU、LSTM、Transformer模型
- **torchvision** - PyTorch视觉库（torch依赖）
- **scikit-learn** - 机器学习库，用于RandomForest、GaussianProcess等模型
- **joblib** - 模型序列化，sklearn依赖

### 深度强化学习
- **stable-baselines3** - 强化学习算法库，用于DQN、PPO、DDPG、SAC
- **gymnasium** - 强化学习环境接口（stable-baselines3依赖）

### 优化算法
- **scipy** - 科学计算库，用于模拟退火、遗传算法、梯度下降
- **bayesian-optimization** - 贝叶斯优化算法

### 通信协议
- **paho-mqtt** - MQTT客户端，用于设备通信
- **requests** - HTTP请求库，用于API通信

### 进度显示
- **tqdm** - 进度条显示，用于训练过程可视化

## 可选依赖

项目还定义了一些可选依赖，可以根据需要安装：

### 开发和调试工具
```bash
pip install -e ".[dev]"
```

包含：
- **ipython** - 交互式Python环境
- **jupyter** - Jupyter Notebook
- **matplotlib** - 数据可视化
- **seaborn** - 统计数据可视化

## 常见问题

### 1. 内置模块依赖错误

**问题**：尝试安装`datetime`、`logging`或`typing`等内置模块时出错

**解决方案**：这些是Python内置模块，不需要单独安装。已从依赖列表中移除。

### 2. PyTorch安装问题

**问题**：PyTorch安装失败或无法使用GPU

**解决方案**：根据您的CUDA版本，使用PyTorch官方安装命令：
```bash
# CPU版本
pip install torch torchvision

# CUDA 11.8版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 版本冲突

**问题**：依赖版本冲突

**解决方案**：使用虚拟环境隔离项目依赖：
```bash
# 创建虚拟环境
python -m venv foshan-env

# 激活虚拟环境
# Windows
foshan-env\Scripts\activate
# Linux/Mac
source foshan-env/bin/activate

# 安装依赖
pip install -e .
```

## 依赖更新说明

最近的依赖更新修复了以下问题：

1. 移除了内置模块依赖：`datetime`、`logging`、`typing`
2. 统一了`pyproject.toml`和`requirements.txt`中的版本范围
3. 添加了所有实际使用的第三方库
4. 确保了依赖间的版本兼容性
5. 添加了详细的注释说明每个依赖的用途
