# CIFAR-10 数据集支持 - 修改说明

本文档记录了为添加 CIFAR-10 数据集支持所进行的所有代码修改。

---

## 修改概览

**目标**：添加 CIFAR-10 数据集支持，通过命令行参数 `--dataset` 选择数据集。

**影响范围**：
- 配置管理
- 命令行参数解析
- 模型定义
- 数据分发
- 主训练脚本

---

## 详细修改

### 1. 配置模块 (`src/config.py`)

**修改内容**：
- 添加 `dataset` 参数（默认：`'fashion-mnist'`）
- 添加 `image_height` 参数（28 for Fashion-MNIST, 32 for CIFAR-10）
- 添加 `image_channels` 参数（1 for Fashion-MNIST, 3 for CIFAR-10）
- 更新 `from_args()` 方法处理数据集参数
- 更新 `__str__()` 方法显示数据集信息

**代码片段**：
```python
# Dataset parameters
self.dataset = "fashion-mnist"  # 'fashion-mnist' or 'cifar-10'
self.image_height = 28  # 28 for Fashion-MNIST, 32 for CIFAR-10
self.image_channels = 1  # 1 for Fashion-MNIST, 3 for CIFAR-10
```

---

### 2. 命令行参数 (`src/utils/helpers.py`)

**修改内容**：
- 添加 `--dataset` 参数，可选值：`fashion-mnist`, `cifar-10`

**代码片段**：
```python
# 数据集参数
parser.add_argument('--dataset', type=str, default='fashion-mnist',
                   choices=['fashion-mnist', 'cifar-10'],
                   help='Dataset to use: fashion-mnist or cifar-10')
```

---

### 3. ResNet 模型 (`src/models/resnet.py`)

**修改内容**：
- 更新文档字符串，明确支持 Fashion-MNIST 和 CIFAR-10
- `in_channel` 参数支持 1（灰度）和 3（RGB）
- 更新函数文档说明输入通道配置

**代码片段**：
```python
def ResNet18(num_classes=10, in_channel=1):
    """
    创建 ResNet18 模型
    
    Args:
        num_classes: 分类数量，默认 10
        in_channel: 输入通道数
            - 1: 灰度图 (Fashion-MNIST/MNIST)
            - 3: RGB 图 (CIFAR-10)
    """
    return ResNet(ResidualBlock, num_classes, in_channel)
```

---

### 4. 分割 ResNet 模型 (`src/models/split_resnet.py`)

**修改内容**：
- `MultiClientNet` 类添加 `input_height` 和 `in_channel` 参数
- `create_multi_client_models()` 方法添加 `input_height` 和 `in_channel` 参数
- 更新 forward 方法使用配置的图像高度和通道数
- 更新文档字符串

**关键变化**：
```python
# 之前
x = x.view(x.shape[0], 1, 28, -1)

# 之后
x = x.view(x.shape[0], self.in_channel, self.input_height, -1)
```

**方法签名**：
```python
@staticmethod
def create_multi_client_models(n_clients=4, input_width=4, feature_dim=256, 
                                hidden_dim=64, num_classes=10, 
                                input_height=28, in_channel=1):
```

---

### 5. 数据分发器 (`src/data/distributor.py`)

**修改内容**：
- `__init__()` 添加 `image_height` 和 `image_channels` 参数
- 更新数据切分逻辑，支持多通道图像
- 移除 `squeeze(1)` 操作，保留通道维度
- 更新 `_create_test_set()` 方法

**关键变化**：
```python
# 之前 (Fashion-MNIST)
image_part = images[:, :, :, start_col:end_col].squeeze(1)  # (batch, 28, width)
curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)

# 之后 (支持多数据集)
image_part = images[:, :, :, start_col:end_col]  # (batch, channels, height, width)
curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)
```

**数据划分说明**：
- **Fashion-MNIST**: `(batch, 1, 28, 28)` → 每个客户端 `(batch, 1, 28, width)` → reshape 为 `(batch, 28*width)`
- **CIFAR-10**: `(batch, 3, 32, 32)` → 每个客户端 `(batch, 3, 32, width)` → reshape 为 `(batch, 3*32*width)`

---

### 6. 主训练脚本 (`test_gpu.py`)

#### 6.1 数据加载

**修改内容**：
- 添加数据集判断逻辑
- CIFAR-10 使用标准数据增强（RandomHorizontalFlip, RandomCrop）
- CIFAR-10 使用标准归一化参数

**代码片段**：
```python
if config.dataset == 'cifar-10':
    # CIFAR-10: 32x32 RGB images, 10 classes
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = datasets.CIFAR10(...)
else:  # fashion-mnist
    trainset = datasets.FashionMNIST(...)
```

#### 6.2 数据分发器初始化

**修改内容**：
- 传递 `image_height` 和 `image_channels` 参数

**代码片段**：
```python
distributor = DataDistributor(
    config.n_clients, trainloader, device, testloader,
    image_height=config.image_height,
    image_channels=config.image_channels
)
```

#### 6.3 模型创建

**修改内容**：
- 使用 `config.image_height` 计算每个客户端的输入宽度
- 传递 `input_height` 和 `in_channel` 参数

**代码片段**：
```python
input_width = config.image_height // config.n_clients

ClientModel, ServerModel = SplitResNet18.create_multi_client_models(
    n_clients=config.n_clients,
    input_width=input_width,
    feature_dim=config.feature_dim,
    hidden_dim=config.hidden_dim,
    num_classes=config.num_classes,
    input_height=config.image_height,
    in_channel=config.image_channels
)
```

---

## 使用方法

### Fashion-MNIST（默认）

```bash
python test_gpu.py
python test_gpu.py --dataset fashion-mnist
```

### CIFAR-10

```bash
python test_gpu.py --dataset cifar-10
python test_gpu.py --dataset cifar-10 --epochs 100 --lr 0.01
```

---

## 数据集对比

| 特性 | Fashion-MNIST | CIFAR-10 |
|------|---------------|----------|
| 图像尺寸 | 28x28 | 32x32 |
| 通道数 | 1 (灰度) | 3 (RGB) |
| 类别数 | 10 | 10 |
| 训练样本 | 60,000 | 50,000 |
| 测试样本 | 10,000 | 10,000 |
| 数据增强 | 无 | RandomHorizontalFlip, RandomCrop |
| 归一化 | (0.5,), (0.5,) | ImageNet 统计量 |

---

## 客户端数据划分示例

### Fashion-MNIST (10 clients)

每个客户端接收：
- 图像切片：28×(28/10) = 28×3 (约 84 像素)
- 展平维度：1×28×3 = 84

### CIFAR-10 (10 clients)

每个客户端接收：
- 图像切片：3×32×(32/10) = 3×32×3 (约 288 像素)
- 展平维度：3×32×3 = 288

---

## 注意事项

1. **首次运行**会自动下载 CIFAR-10 数据集到 `datasets/cifar-10/` 目录

2. **模型参数数量**：
   - CIFAR-10 使用 3 通道输入，与 Fashion-MNIST 的 1 通道相比，第一卷积层参数数量增加 3 倍
   - 其余层参数数量保持不变

3. **训练时间**：
   - CIFAR-10 数据量更大（50k vs 60k），但图像尺寸接近
   - 主要差异来自数据增强（RandomHorizontalFlip, RandomCrop）

4. **推荐超参数**：
   - Fashion-MNIST: `--lr 0.001`
   - CIFAR-10: `--lr 0.01`（可能需要调整）

5. **数据预处理**：
   - CIFAR-10 使用标准归一化参数（基于 ImageNet 统计量）
   - Fashion-MNIST 使用简单归一化 (0.5, 0.5)

---

## 技术细节

### ResNet18 适配

原始 ResNet18 针对 ImageNet (224×224) 设计，本项目已针对小图像优化：
- 第一卷积层：7×7 stride=2 → 3×3 stride=1
- 移除第一个 MaxPool（图像太小不需要）
- 使用 AdaptiveAvgPool2d 适配任意输入尺寸

### 垂直分割实现

**按列切分**：每个客户端处理图像的一部分列（width 维度）

```
原始图像 (28×28 或 32×32):
┌─────────────────┐
│ C0 │ C1 │ C2 │...│  ← 按列切分
└─────────────────┘

C0, C1, C2... 分发给不同客户端
每个客户端独立处理自己的列切片
```

**优势**：
- 模拟真实场景中不同机构拥有不同特征维度
- 客户端模型可以独立训练
- 服务器聚合所有客户端特征进行分类

---

## 测试验证

建议测试：

```bash
# 快速测试 Fashion-MNIST
python test_gpu.py --dataset fashion-mnist --epochs 10 --clients 5

# 快速测试 CIFAR-10
python test_gpu.py --dataset cifar-10 --epochs 10 --clients 5

# 完整训练
python test_gpu.py --dataset cifar-10 --epochs 100 --encryption tenseal
```

---

**修改完成时间**：2026-04-10
