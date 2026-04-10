"""
Encryption Transmission Base Class
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple
import torch
import time


@dataclass
class TransmissionConfig:
    """Transmission configuration"""
    method: str = 'plaintext'  # 'plaintext', 'paillier', 'tenseal'
    key_size: int = 2048       # Key size (Paillier)
    poly_modulus_degree: int = 8192  # Polynomial modulus degree (CKKS)
    coeff_mod_bit_sizes: list = None  # Coefficient modulus bit sizes (CKKS)
    
    def __post_init__(self):
        if self.coeff_mod_bit_sizes is None:
            self.coeff_mod_bit_sizes = [60, 40, 40, 60]


class BaseTransmission(ABC):
    """Encryption transmission base class"""
    
    def __init__(self, config: TransmissionConfig = None):
        self.config = config or TransmissionConfig()
        self.encrypt_time = 0.0
        self.decrypt_time = 0.0
        self.transfer_time = 0.0
    
    @abstractmethod
    def encrypt_tensor(self, tensor: torch.Tensor) -> Any:
        """Encrypt tensor
        
        Args:
            tensor: Tensor to encrypt
            
        Returns:
            Encrypted data
        """
        pass
    
    @abstractmethod
    def decrypt_tensor(self, encrypted_data: Any) -> torch.Tensor:
        """Decrypt tensor
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted tensor
        """
        pass
    
    def transmit(self, tensor: torch.Tensor, simulate_delay: float = 0.0) -> Tuple[torch.Tensor, dict]:
        """Transmit tensor (encrypt -> transmit -> decrypt)
        
        Args:
            tensor: Tensor to transmit
            simulate_delay: Simulated network delay (seconds)
            
        Returns:
            (decrypted tensor, timing statistics dict)
        """
        timings = {}
        
        # Encrypt
        t0 = time.time()
        encrypted = self.encrypt_tensor(tensor)
        timings['encrypt_time'] = time.time() - t0
        self.encrypt_time += timings['encrypt_time']
        
        # Simulate transmission delay
        if simulate_delay > 0:
            time.sleep(simulate_delay)
        timings['transfer_time'] = simulate_delay
        self.transfer_time += timings['transfer_time']
        
        # Decrypt
        t0 = time.time()
        decrypted = self.decrypt_tensor(encrypted)
        timings['decrypt_time'] = time.time() - t0
        self.decrypt_time += timings['decrypt_time']
        
        timings['total_time'] = timings['encrypt_time'] + timings['transfer_time'] + timings['decrypt_time']
        
        return decrypted, timings
    
    def get_stats(self) -> dict:
        """Get statistics"""
        return {
            'encrypt_time': self.encrypt_time,
            'decrypt_time': self.decrypt_time,
            'transfer_time': self.transfer_time,
            'total_time': self.encrypt_time + self.decrypt_time + self.transfer_time
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.encrypt_time = 0.0
        self.decrypt_time = 0.0
        self.transfer_time = 0.0
