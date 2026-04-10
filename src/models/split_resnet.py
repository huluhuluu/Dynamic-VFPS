"""
ResNet18 分割模型 - 用于垂直联邦学习

提供两种分割方案：
1. 单客户端分割：ResNet18 在客户端，分类头在服务器
2. 多客户端垂直分割：每个客户端处理图像的一部分列

支持数据集：
- Fashion-MNIST/MNIST: 28x28 单通道图像
- CIFAR-10: 32x32 三通道图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResidualBlock


class ClientNet(nn.Module):
    """
    客户端网络 - 单客户端版本
    
    使用 ResNet18 提取特征
    参考 vflweight 的设计
    """
    
    def __init__(self, feature_dim=1152):
        super(ClientNet, self).__init__()
        from .resnet import ResNet
        self.resnet18 = ResNet(ResidualBlock, num_classes=feature_dim, in_channel=1)
        self.feature_dim = feature_dim

    def forward(self, x):
        # 输入可能是 (batch, 784) 或 (batch, 28, 28) 或 (batch, 1, 28, 28)
        if x.dim() == 2:
            # (batch, 784) -> (batch, 1, 28, 28)
            x = x.view(x.shape[0], 1, 28, -1)
        elif x.dim() == 3:
            # (batch, 28, 28) -> (batch, 1, 28, 28)
            x = x.unsqueeze(1)
        # else: 已经是 (batch, 1, 28, 28)
        
        x = self.resnet18(x)
        return x  # 输出: (batch, feature_dim)


class ServerNet(nn.Module):
    """
    服务器网络 - 单客户端版本
    
    接收客户端特征，完成分类
    """
    
    def __init__(self, input_dim=1152, hidden_dim=64, num_classes=10):
        super(ServerNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 输出: (batch, num_classes)


class MultiClientNet(nn.Module):
    """
    客户端网络 - 多客户端垂直分割版本
    
    参考 vflweight 的设计：
    - 垂直分割按列进行，每个客户端收到 heightx(width) 的图像
    - 直接使用完整的 ResNet18
    - 输入: (batch, height, width) 或 (batch, height*width)
    - 支持不等宽输入（不同客户端可能有不同宽度）
    """
    
    def __init__(self, input_width=4, feature_dim=256, input_height=28, in_channel=1):
        super(MultiClientNet, self).__init__()
        self.input_width = input_width  # 仅用于参考，实际宽度由输入决定
        self.input_height = input_height
        self.in_channel = in_channel
        self.feature_dim = feature_dim
        
        # 直接使用完整的 ResNet18
        from .resnet import ResNet, ResidualBlock
        self.resnet18 = ResNet(ResidualBlock, num_classes=feature_dim, in_channel=in_channel)
    
    def forward(self, x):
        # 动态计算输入宽度（支持不等宽输入）
        # 输入: (batch, height*width) 或 (batch, in_channel, height, width)
        batch_size = x.shape[0]
        if x.dim() == 2:
            # (batch, height*width) -> (batch, in_channel, height, width)
            # 自动计算宽度
            total_elements = x.shape[1]
            width = total_elements // (self.in_channel * self.input_height)
            x = x.view(batch_size, self.in_channel, self.input_height, width)
        else:
            # 已经是 (batch, in_channel, height, width) 格式
            pass
        x = self.resnet18(x)
        return x


class MultiClientServerNet(nn.Module):
    """
    服务器网络 - 多客户端版本
    
    聚合所有客户端特征后分类
    """
    
    def __init__(self, n_clients=4, feature_dim=256, hidden_dim=64, num_classes=10):
        super(MultiClientServerNet, self).__init__()
        total_feature_dim = feature_dim * n_clients
        self.fc1 = nn.Linear(total_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, *client_features):
        # 拼接所有客户端特征
        x = torch.cat(client_features, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SplitResNet18:
    """
    ResNet18 分割模型工厂类
    
    提供创建不同分割方案的静态方法
    """
    
    @staticmethod
    def create_single_client_models(feature_dim=1152, hidden_dim=64, num_classes=10):
        """
        创建单客户端分割模型
        
        Args:
            feature_dim: 特征维度 (默认 1152，参考 vflweight)
            hidden_dim: 服务器隐藏层维度
            num_classes: 分类数
        
        Returns:
            (ClientModelClass, ServerModelClass) 元组 - 返回类而不是实例
        """
        # 返回类，让用户可以实例化多次
        class ClientModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                from .resnet import ResNet, ResidualBlock
                self.resnet18 = ResNet(ResidualBlock, num_classes=feature_dim, in_channel=1)
                self.feature_dim = feature_dim

            def forward(self, x):
                if x.dim() == 2:
                    x = x.view(x.shape[0], 1, 28, -1)
                elif x.dim() == 3:
                    x = x.unsqueeze(1)
                x = self.resnet18(x)
                return x
        
        class ServerModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(feature_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, num_classes)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                # 添加 LogSoftmax 以配合 NLLLoss
                return F.log_softmax(x, dim=1)
        
        return ClientModelClass, ServerModelClass
    
    @staticmethod
    def create_multi_client_models(n_clients=4, input_width=4, feature_dim=256, 
                                    hidden_dim=64, num_classes=10, input_height=28, in_channel=1):
        """
        创建多客户端垂直分割模型
        
        参考 vflweight 的设计：
        - 垂直分割按列进行
        - 每个客户端收到 heightx(width) 的图像
        - 高度固定，宽度可变
        
        Args:
            n_clients: 客户端数量
            input_width: 每个客户端处理的图像宽度
            feature_dim: 每个客户端的特征维度
            hidden_dim: 服务器隐藏层维度
            num_classes: 分类数
            input_height: 输入图像高度 (28 for Fashion-MNIST, 32 for CIFAR-10)
            in_channel: 输入通道数 (1 for Fashion-MNIST, 3 for CIFAR-10)
        
        Returns:
            (ClientModelClass, ServerModelClass) 元组
        """
        from .resnet import ResNet, ResidualBlock
        
        class ClientModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_width = input_width
                self.input_height = input_height
                self.in_channel = in_channel
                # 直接使用 ResNet18
                self.resnet18 = ResNet(ResidualBlock, num_classes=feature_dim, in_channel=in_channel)
            
            def forward(self, x):
                # 动态计算输入宽度（支持不等宽输入）
                # 输入: (batch, height*width) -> 输出: (batch, in_channel, height, width)
                batch_size = x.shape[0]
                if x.dim() == 2:
                    total_elements = x.shape[1]
                    width = total_elements // (self.in_channel * self.input_height)
                    x = x.view(batch_size, self.in_channel, self.input_height, width)
                x = self.resnet18(x)
                return x
        
        class ServerModelClass(nn.Module):
            def __init__(self):
                super().__init__()
                total_feature_dim = feature_dim * n_clients
                self.fc1 = nn.Linear(total_feature_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, num_classes)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)
        
        return ClientModelClass, ServerModelClass
        
        return ClientModelClass, ServerModelClass


if __name__ == '__main__':
    # 测试单客户端版本
    print("=== 单客户端版本测试 ===")
    client_model, server_model = SplitResNet18.create_single_client_models()
    
    x = torch.randn(64, 784)  # 模拟输入
    features = client_model(x)
    output = server_model(features)
    
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    
    # 测试多客户端版本
    print("\n=== 多客户端版本测试 ===")
    client_model, server_model = SplitResNet18.create_multi_client_models(n_clients=4)
    
    # 模拟 4 个客户端的输入
    client_inputs = [torch.randn(64, 196) for _ in range(4)]  # 每个客户端 196=7*28
    
    client_features = [client_model(inp) for inp in client_inputs]
    output = server_model(*client_features)
    
    print(f"Each client input shape: {client_inputs[0].shape}")
    print(f"Each client features shape: {client_features[0].shape}")
    print(f"Output shape: {output.shape}")
