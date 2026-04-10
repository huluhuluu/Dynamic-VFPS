"""
Plaintext Transmission (No Encryption)
"""

import torch
from typing import Any
from .base import BaseTransmission, TransmissionConfig


class PlaintextTransmission(BaseTransmission):
    """Plaintext transmission - no encryption, direct transmission"""
    
    def __init__(self, config: TransmissionConfig = None):
        super().__init__(config)
        self.method = 'plaintext'
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Any:
        """Plaintext transmission requires no encryption, return directly"""
        return tensor.clone()
    
    def decrypt_tensor(self, encrypted_data: Any) -> torch.Tensor:
        """Plaintext transmission requires no decryption, return directly"""
        return encrypted_data
