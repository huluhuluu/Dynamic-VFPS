"""
数据分发器 - 垂直联邦学习
"""

import random
import torch
from typing import List, Dict, Tuple, Any


class DataDistributor:
    """数据分发器 - 用于垂直联邦学习
    
    基于 vflweight 设计：按列切分
    - 每个客户端接收 heightx(width) 图像
    - 高度固定，宽度可变
    
    支持：
    - Fashion-MNIST: 28x28 单通道
    - CIFAR-10: 32x32 三通道
    """
    
    def __init__(self, n_clients: int, data_loader, device, test_loader=None, 
                 image_height: int = 28, image_channels: int = 1):
        """初始化数据分发器
        
        Args:
            n_clients: 客户端数量
            data_loader: 训练数据加载器
            device: 计算设备
            test_loader: 测试数据加载器（可选）
            image_height: 图像高度 (28 for Fashion-MNIST, 32 for CIFAR-10)
            image_channels: 图像通道数 (1 for Fashion-MNIST, 3 for CIFAR-10)
        """
        self.n_clients = n_clients
        self.device = device
        self.image_height = image_height
        self.image_channels = image_channels
        
        # 垂直划分训练数据（按列）
        self.data_pointer: List[Dict[str, torch.Tensor]] = []
        self.labels: List[torch.Tensor] = []
        
        for images, labels in data_loader:
            images = images.to(device)
            # 对于 CIFAR-10，图像已经是三通道的，不需要额外处理
            width = images.shape[-1] // n_clients
            
            curr_data = {}
            for i in range(n_clients):
                start_col = i * width
                end_col = start_col + width
                # 按列分割: 
                # Fashion-MNIST: (batch, 1, 28, 28) -> (batch, 28, width) -> (batch, 28*width)
                # CIFAR-10: (batch, 3, 32, 32) -> (batch, 3, 32, width) -> (batch, 3*32*width)
                image_part = images[:, :, :, start_col:end_col]  # (batch, channels, height, width)
                curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)
            
            self.data_pointer.append(curr_data)
            self.labels.append(labels)
        
        # 创建测试集（使用真正的测试数据）
        self.test_set = self._create_test_set(test_loader)
        
        # 训练子数据
        self.subdata: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]] = []
    
    def _create_test_set(self, test_loader) -> List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """使用真正的测试数据创建测试集
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            测试集列表
        """
        if test_loader is None:
            return []
        
        test_set = []
        for images, labels in test_loader:
            images = images.to(self.device)
            width = images.shape[-1] // self.n_clients
            
            curr_data = {}
            for i in range(self.n_clients):
                start_col = i * width
                end_col = start_col + width
                image_part = images[:, :, :, start_col:end_col]
                curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)
            
            test_set.append((curr_data, labels))
        
        return test_set
    
    def generate_subdata(self, prob: float = 0.2):
        """生成训练子数据
        
        Args:
            prob: 采样概率
        """
        self.subdata = []
        for idx, (data_ptr, label) in enumerate(zip(self.data_pointer, self.labels)):
            if random.random() <= prob:
                self.subdata.append((idx, data_ptr, label))
    
    def generate_estimate_subdata(self, n_samples: int = 50) -> List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]]:
        """生成用于互信息估计的子数据
        
        Args:
            n_samples: 采样数量
            
        Returns:
            子数据列表
        """
        n_samples = min(n_samples, len(self.data_pointer))
        indices = random.sample(range(len(self.data_pointer)), n_samples)
        return [(idx, self.data_pointer[idx], self.labels[idx]) for idx in indices]
    
    @property
    def n_batches(self) -> int:
        """批次数量"""
        return len(self.data_pointer)
