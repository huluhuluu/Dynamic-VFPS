"""
Split Neural Network for Vertical Federated Learning
"""

import time
import torch
import numpy as np
from torch import nn
from typing import Dict, List, Tuple, Any

from src.config import Config
from src.communication.estimator import CommunicationEstimator
from src.utils.helpers import digamma


class SplitNN:
    """Split neural network for VFL with MI-based dynamic client selection"""
    
    def __init__(self, models, config: Config, optimizers, 
                 comm_estimator: CommunicationEstimator, device):
        """Initialize split neural network
        
        Args:
            models: Model dictionary
            config: Training configuration
            optimizers: Optimizer dictionary
            comm_estimator: Communication estimator
            device: Computation device
        """
        self.models = models
        self.config = config
        self.optimizers = optimizers
        self.comm_estimator = comm_estimator
        self.device = device
        
        # Client selection state
        self.selected = {f"client_{i}": True for i in range(config.n_clients)}
        
        # MI estimation scores
        self.scores: Dict[str, float] = {}
        
        # Padding cache
        self.latest: Dict[str, torch.Tensor] = {}
    
    # -------------------------------------------------------------------------
    # Forward Propagation
    # -------------------------------------------------------------------------
    
    def predict(self, data_ptr: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float, List[torch.Tensor]]:
        """Forward propagation (plaintext transmission for training)
        
        Args:
            data_ptr: Data pointer
            
        Returns:
            (pred, comm_time, client_outputs): prediction, communication time, client outputs
        """
        client_outputs = []
        client_times = []
        
        for i in range(self.config.n_clients):
            client_id = f"client_{i}"
            
            if self.selected[client_id]:
                # Client forward pass
                output = self.models[client_id](data_ptr[client_id])
                
                # Plaintext transmission estimation
                t = self.comm_estimator.estimate_plaintext(output)
                client_times.append(t)
                
                # Save output for gradient transmission estimation
                client_outputs.append(output)
                
                # Update padding cache
                self._update_padding_cache(client_id, output)
            else:
                # Use padding
                padding = self._get_padding(client_id, data_ptr)
                client_outputs.append(padding)
        
        # Server forward pass
        server_input = torch.cat(client_outputs, dim=1)
        pred = self.models["server"](server_input)
        
        return pred, max(client_times) if client_times else 0.0, client_outputs
    
    def _update_padding_cache(self, client_id: str, output: torch.Tensor):
        """Update padding cache"""
        self.latest[client_id] = output.detach().clone()
    
    def _get_padding(self, client_id: str, data_ptr: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get padding tensor"""
        batch_size = data_ptr[client_id].size(0)
        
        if self.config.padding_method == "latest" and client_id in self.latest:
            latest = self.latest[client_id]
            # Use latest if batch size matches
            if latest.size(0) == batch_size:
                return latest
        
        # Default: zeros padding
        return torch.zeros(batch_size, self.config.feature_dim, device=self.device)
    
    # -------------------------------------------------------------------------
    # Training Step
    # -------------------------------------------------------------------------
    
    def train_step(self, data_ptr: Dict[str, torch.Tensor], target: torch.Tensor, 
                   local_epochs: int = 1) -> Tuple[float, float, float]:
        """Single training step with local iterations
        
        Args:
            data_ptr: Data pointer
            target: Labels
            local_epochs: Number of local iterations
        
        Returns:
            (loss, train_time, comm_time): average loss, training time, communication time
        """
        total_loss = 0.0
        total_comm_time = 0.0
        
        for local_iter in range(local_epochs):
            iter_start = time.time()
            
            # Zero gradients
            for opt in self.optimizers.values():
                opt.zero_grad()
            
            # Forward propagation
            pred, fwd_comm_time, client_outputs = self.predict(data_ptr)
            loss = nn.NLLLoss()(pred, target)
            total_loss += loss.item()
            
            # Backward propagation
            loss.backward()
            
            # Accumulate communication time for this round (forward + backward, gradient size same as output)
            total_comm_time += fwd_comm_time * 2
            
            # Update parameters
            for client_id, opt in self.optimizers.items():
                if client_id == "server":
                    continue
                if self.selected.get(client_id, True):
                    opt.step()
            self.optimizers["server"].step()
        
        # Training time (computation only, excluding communication)
        train_time = time.time() - iter_start
        
        return total_loss / local_epochs, train_time, total_comm_time
    
    # -------------------------------------------------------------------------
    # Mutual Information Estimation
    # -------------------------------------------------------------------------
    
    def estimate_mi_cuda(self, subdata: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]]) -> float:
        """CUDA-based KNN mutual information estimation
        
        Note: Communication time is calculated in group_testing.
        This method only computes MI value.
        
        Args:
            subdata: Subset data list
            
        Returns:
            Mutual information value
        """
        if not subdata:
            return 0.0
        
        # Batch feature extraction (using all samples)
        features_list = []
        targets = []
        
        with torch.no_grad():
            for _, data_ptr, target in subdata:
                # target is the label for entire batch
                batch_size = target.size(0) if isinstance(target, torch.Tensor) else len(target)
                
                # For each sample in the batch
                for sample_idx in range(batch_size):
                    # Aggregate features from selected clients
                    combined_feat = []
                    for i in range(self.config.n_clients):
                        client_id = f"client_{i}"
                        if self.selected[client_id]:
                            # Take single sample
                            sample_data = data_ptr[client_id][sample_idx:sample_idx+1]
                            feat = self.models[client_id](sample_data)
                            combined_feat.append(feat[0])
                    
                    if combined_feat:
                        features_list.append(torch.cat(combined_feat))
                    
                    # Get corresponding target
                    t = target[sample_idx].item() if isinstance(target, torch.Tensor) else target[sample_idx]
                    targets.append(t)
        
        n_samples = len(features_list)
        if n_samples == 0:
            return 0.0
        
        # Compute distance matrix
        features = torch.stack(features_list)  # [n_samples, feature_dim * n_selected]
        dist_matrix = torch.cdist(features, features)
        dist_matrix.fill_diagonal_(float('inf'))  # Exclude self
        
        # Compute MI
        targets_tensor = torch.tensor(targets, device=self.device)
        mi = 0.0
        
        for idx in range(n_samples):
            target = targets[idx]
            
            # In-class distance
            class_mask = (targets_tensor == target)
            Nq = class_mask.sum().item()
            
            class_indices = torch.where(class_mask)[0]
            class_dists = dist_matrix[idx, class_indices]
            
            # k-th nearest neighbor distance
            k = min(self.config.k_nn, len(class_dists))
            if k == 0:
                continue
            
            rho_k = torch.kthvalue(class_dists, k).values.item()
            
            # Count mq
            mq = (dist_matrix[idx] < rho_k).sum().item()
            
            # MI formula
            if mq > 0:
                mi += digamma(n_samples) - digamma(Nq) + digamma(k) - digamma(mq)
        
        return mi / n_samples
    
    # -------------------------------------------------------------------------
    # Group Testing
    # -------------------------------------------------------------------------
    
    def group_testing(self, estimate_subdata: List[Tuple[int, Dict[str, torch.Tensor], torch.Tensor]], 
                      n_tests: int) -> Tuple[Dict[str, float], float, float]:
        """Group testing for client selection
        
        Process:
        1. All clients send encrypted data to server (one communication)
        2. Server performs n_tests group tests locally (no extra communication)
        
        Args:
            estimate_subdata: Data subset for estimation
            n_tests: Number of group tests
            
        Returns:
            (scores, mi_comm_time, mi_comp_time): client scores, communication time, computation time
        """
        self.scores = {f"client_{i}": 0.0 for i in range(self.config.n_clients)}
        
        # 1. Calculate encrypted communication time for all data
        # Each batch: all clients send in parallel, take max
        # Total time: sum of all batch communication times
        batch_comm_times = []
        for _, data_ptr, _ in estimate_subdata:
            client_times = []
            for i in range(self.config.n_clients):
                client_id = f"client_{i}"
                t = self.comm_estimator.estimate_encrypted(data_ptr[client_id])
                client_times.append(t)
            # Batch communication time = max (parallel transmission)
            batch_comm_times.append(max(client_times) if client_times else 0.0)
        
        # Total communication time = sum of all batch times
        mi_comm_time = sum(batch_comm_times)
        
        mi_start = time.time()
        # 2. Perform n_tests group tests locally (no extra communication)
        for _ in range(n_tests):
            # Randomly generate test group
            test_group = self._generate_test_group()
            
            # Temporarily set selected state
            original_selected = self.selected.copy()
            self.selected = {f"client_{i}": (f"client_{i}" in test_group) 
                           for i in range(self.config.n_clients)}
            
            # Compute MI (local computation, no communication)
            mi = self.estimate_mi_cuda(estimate_subdata)
            
            # Accumulate scores
            for client_id in test_group:
                self.scores[client_id] += mi
            
            # Restore selected state
            self.selected = original_selected
        
        # Average scores and select
        self.scores = {k: v / n_tests for k, v in self.scores.items()}
        self._select_top_clients()
        
        return self.scores, mi_comm_time, time.time() - mi_start
    
    def _generate_test_group(self, p: float = 0.5) -> List[str]:
        """Randomly generate test group
        
        Args:
            p: Selection probability
            
        Returns:
            Test group client ID list
        """
        test_group = []
        while len(test_group) < 1:
            for i in range(self.config.n_clients):
                if np.random.rand() < p:
                    test_group.append(f"client_{i}")
        return test_group
    
    def _select_top_clients(self):
        """Select top-scored clients"""
        sorted_clients = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        
        self.selected = {f"client_{i}": False for i in range(self.config.n_clients)}
        for client_id, _ in sorted_clients[:self.config.n_selected]:
            self.selected[client_id] = True
