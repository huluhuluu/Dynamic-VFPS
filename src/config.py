"""
Training Configuration Module
"""

import argparse


class Config:
    """Training configuration"""
    
    def __init__(self):
        # Training parameters
        self.epochs = 50
        self.learning_rate = 0.001
        self.batch_size = 256
        self.local_epochs = 1  # Local iterations per batch
        self.subset_update_prob = 0.2
        
        # Client parameters
        self.n_clients = 10
        self.n_selected = 6
        
        # MI estimation parameters
        self.n_tests = 5
        self.k_nn = 3
        
        # Communication parameters
        self.bandwidth_mbps = 300
        self.padding_method = "zeros"
        
        # Encryption method
        self.encryption = "plaintext"
        
        # Dataset parameters
        self.dataset = "fashion-mnist"  # 'fashion-mnist' or 'cifar-10'
        self.image_height = 28  # 28 for Fashion-MNIST, 32 for CIFAR-10
        self.image_channels = 1  # 1 for Fashion-MNIST, 3 for CIFAR-10
        
        # Model parameters
        self.feature_dim = 256
        self.hidden_dim = 128
        self.num_classes = 10
        
        # Evaluation parameters
        self.eval_every_steps = 10
        self.estimate_samples = 50
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create configuration from command line arguments
        
        Args:
            args: Command line arguments
            
        Returns:
            Config instance
        """
        config = cls()
        config.epochs = args.epochs
        config.learning_rate = args.lr
        config.batch_size = args.batch_size
        config.local_epochs = args.local_epochs
        config.n_clients = args.clients
        config.n_selected = args.selected
        config.n_tests = args.n_tests
        config.k_nn = args.k_nn
        config.encryption = args.encryption
        config.bandwidth_mbps = args.bandwidth
        
        # Dataset configuration
        config.dataset = args.dataset
        if args.dataset == 'cifar-10':
            config.image_height = 32
            config.image_channels = 3
        else:  # fashion-mnist
            config.image_height = 28
            config.image_channels = 1
        
        return config
    
    def __str__(self) -> str:
        return (
            f"Dataset: {self.dataset}, Epochs: {self.epochs}, Local epochs: {self.local_epochs}, "
            f"Clients: {self.n_clients}/{self.n_selected}, "
            f"Encryption: {self.encryption}, Bandwidth: {self.bandwidth_mbps} Mbps"
        )
