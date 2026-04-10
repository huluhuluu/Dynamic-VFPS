"""
Data Distributor for Vertical Federated Learning
"""

import random
import torch
from typing import List, Dict, Tuple, Any


class DataDistributor:
    """Distribute data vertically across clients
    
    Column-wise split: each client receives height x width slice.
    Supports Fashion-MNIST (28x28x1) and CIFAR-10 (32x32x3).
    """
    
    def __init__(self, n_clients: int, data_loader, device, test_loader=None, 
                 image_height: int = 28, image_channels: int = 1):
        """Initialize data distributor
        
        Args:
            n_clients: Number of clients
            data_loader: Training data loader
            device: Computation device
            test_loader: Test data loader (optional)
            image_height: Image height (28 for Fashion-MNIST, 32 for CIFAR-10)
            image_channels: Number of channels (1 for Fashion-MNIST, 3 for CIFAR-10)
        """
        self.n_clients = n_clients
        self.device = device
        self.image_height = image_height
        self.image_channels = image_channels
        
        # Calculate width for each client (ensure complete split without data loss)
        image_width = image_height
        base_width = image_width // n_clients
        remainder = image_width % n_clients
        
        # Distribute remainder: first 'remainder' clients get 1 extra column
        self.client_widths = []
        for i in range(n_clients):
            client_width = base_width + (1 if i < remainder else 0)
            self.client_widths.append(client_width)
        
        # Verify total width
        total_width = sum(self.client_widths)
        assert total_width == image_width, f"Width allocation error: {total_width} != {image_width}"
        
        # Split training data vertically (by columns)
        self.data_pointer: List[Dict[str, torch.Tensor]] = []
        self.labels: List[torch.Tensor] = []
        
        for images, labels in data_loader:
            images = images.to(device)
            
            curr_data = {}
            start_col = 0
            for i in range(n_clients):
                client_width = self.client_widths[i]
                end_col = start_col + client_width
                image_part = images[:, :, :, start_col:end_col]
                curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)
                start_col = end_col
            
            self.data_pointer.append(curr_data)
            self.labels.append(labels)
        
        # Create test set
        self.test_set = self._create_test_set(test_loader)
        
        # Training subset
        self.subdata: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]] = []
    
    def _create_test_set(self, test_loader) -> List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """Create test set using test data loader
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Test set list
        """
        if test_loader is None:
            return []
        
        test_set = []
        for images, labels in test_loader:
            images = images.to(self.device)
            
            curr_data = {}
            start_col = 0
            for i in range(self.n_clients):
                client_width = self.client_widths[i]
                end_col = start_col + client_width
                image_part = images[:, :, :, start_col:end_col]
                curr_data[f"client_{i}"] = image_part.reshape(images.size(0), -1)
                start_col = end_col
            
            test_set.append((curr_data, labels))
        
        return test_set
    
    def generate_subdata(self, prob: float = 0.2):
        """Generate training subset
        
        Args:
            prob: Sampling probability
        """
        self.subdata = []
        for idx, (data_ptr, label) in enumerate(zip(self.data_pointer, self.labels)):
            if random.random() <= prob:
                self.subdata.append((idx, data_ptr, label))
    
    def generate_estimate_subdata(self, n_samples: int = 50) -> List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]]:
        """Generate subset for MI estimation
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Subset list
        """
        n_samples = min(n_samples, len(self.data_pointer))
        indices = random.sample(range(len(self.data_pointer)), n_samples)
        return [(idx, self.data_pointer[idx], self.labels[idx]) for idx in indices]
    
    @property
    def n_batches(self) -> int:
        """Number of batches"""
        return len(self.data_pointer)
    
    def verify_split(self) -> bool:
        """Verify split correctness (total width equals original image width)
        
        Returns:
            True if correct
        """
        image_width = self.image_height
        total_width = sum(self.client_widths)
        
        if total_width == image_width:
            print(f"✓ Split verification passed: {total_width} == {image_width}")
            print(f"  Client width distribution: {self.client_widths}")
            return True
        else:
            print(f"✗ Split verification failed: {total_width} != {image_width}")
            return False
