#!/usr/bin/env python3
"""
Dynamic-VFPS GPU Test Script
MI-based Dynamic Participant Selection for Vertical Federated Learning

Usage:
    python test_gpu.py                                    # default parameters
    python test_gpu.py --epochs 50 --clients 10          # custom parameters
    python test_gpu.py --encryption paillier             # use Paillier encryption
    python test_gpu.py --mi-mode static                  # static client selection
    python test_gpu.py --help                            # show all parameters
"""

import sys
sys.path.append('./')

import random
import torch
from torchvision import datasets, transforms
from torch import optim

# Import refactored modules
from src.config import Config
from src.utils.helpers import parse_args, get_device
from src.communication.estimator import CommunicationEstimator
from src.data.distributor import DataDistributor
from src.models.split_resnet import SplitResNet18
from src.splitnn import SplitNN
from src.evaluation import evaluate


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    config = Config.from_args(args)
    
    # Device
    device = get_device()
    if torch.cuda.is_available():
        print(f"[INFO] Device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print(f"[INFO] Device: CPU")
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Configuration: {config}")
    print(f"{'='*60}\n")
    
    # Create communication estimator
    comm_estimator = CommunicationEstimator(
        bandwidth_mbps=config.bandwidth_mbps,
        encryption=config.encryption
    )
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
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
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
        trainset = datasets.CIFAR10(
            root='./datasets/cifar-10',
            download=True,
            train=True,
            transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
        
        testset = datasets.CIFAR10(
            root='./datasets/cifar-10',
            download=True,
            train=False,
            transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)
        
        print(f"Dataset: CIFAR-10")
        print(f"  Train: {len(trainset)} samples, {len(trainloader)} batches")
        print(f"  Test:  {len(testset)} samples, {len(testloader)} batches")
        print(f"  Image size: 32x32x3")
        
    else:  # fashion-mnist
        trainset = datasets.FashionMNIST(
            root='./datasets/fashion_mnist',
            download=True,
            train=True,
            transform=transform
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
        
        testset = datasets.FashionMNIST(
            root='./datasets/fashion_mnist',
            download=True,
            train=False,
            transform=transform
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)
        
        print(f"Dataset: Fashion-MNIST")
        print(f"  Train: {len(trainset)} samples, {len(trainloader)} batches")
        print(f"  Test:  {len(testset)} samples, {len(testloader)} batches")
        print(f"  Image size: 28x28x1")
    
    # Data distribution
    distributor = DataDistributor(
        config.n_clients, trainloader, device, testloader,
        image_height=config.image_height,
        image_channels=config.image_channels
    )
    print(f"Data distributed: {distributor.n_batches} batches, {config.n_clients} clients")
    
    # Verify split correctness
    distributor.verify_split()
    
    # Create models
    torch.manual_seed(0)
    
    # Use client width distribution from distributor
    # Note: Different clients may have different input widths
    # But ResNet can handle variable input sizes, and final feature dimension is the same
    avg_input_width = sum(distributor.client_widths) // len(distributor.client_widths)
    
    print(f"Client width distribution: {distributor.client_widths}")
    print(f"Average input width per client: {avg_input_width}")
    
    ClientModel, ServerModel = SplitResNet18.create_multi_client_models(
        n_clients=config.n_clients,
        input_width=avg_input_width,  # 使用平均宽度
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        input_height=config.image_height,
        in_channel=config.image_channels
    )
    
    models = {f"client_{i}": ClientModel().to(device) for i in range(config.n_clients)}
    models["server"] = ServerModel().to(device)
    
    client_params = sum(p.numel() for p in models["client_0"].parameters())
    server_params = sum(p.numel() for p in models["server"].parameters())
    print(f"Model: Client {client_params:,} params, Server {server_params:,} params")
    
    # Create optimizers
    optimizers = {f"client_{i}": optim.SGD(models[f"client_{i}"].parameters(), 
                                           lr=config.learning_rate, momentum=0.9)
                  for i in range(config.n_clients)}
    optimizers["server"] = optim.SGD(models["server"].parameters(), 
                                     lr=config.learning_rate, momentum=0.9)
    
    # Create SplitNN
    splitnn = SplitNN(models, config, optimizers, comm_estimator, device)
    
    # Training loop
    print(f"\n{'='*60}")
    print("[Training Started]")
    print(f"{'='*60}")
    print(f"MI Mode: {args.mi_mode}")
    
    total_train_time = 0.0
    total_mi_compute_time = 0.0  # MI computation time
    total_mi_comm_time = 0.0     # MI communication time
    total_comm_time = 0.0        # Model training communication time
    global_step = 0
    
    # Print data dimension info (only once at first step)
    dim_info_printed = False
    
    # Static mode: Select clients once before training
    if args.mi_mode == 'static':
        print(f"\n[Static MI Mode]")
        print(f"  MI data ratio: {args.mi_ratio}")
        
        # 1. Sample mi_ratio proportion of training data for MI computation
        n_mi_batches = int(len(distributor.data_pointer) * args.mi_ratio)
        mi_indices = random.sample(range(len(distributor.data_pointer)), n_mi_batches)
        mi_data = [(idx, distributor.data_pointer[idx], distributor.labels[idx]) 
                   for idx in mi_indices]
        print(f"  MI batches: {n_mi_batches}")
        
        # 2. Perform group testing to select clients using all MI data
        scores, mi_comm_time, mi_compute_time = splitnn.group_testing(mi_data, config.n_tests)
        
        selected = [k for k, v in splitnn.selected.items() if v]
        print(f"  Selected clients: {selected}")
        print(f"  MI total time: {mi_compute_time + mi_comm_time:.2f}s")
        print(f"  MI compute time: {mi_compute_time:.2f}s")
        print(f"  MI comm time: {mi_comm_time:.4f}s ({n_mi_batches} batches)")
        print(f"\n  Clients fixed for all training steps")
        
        total_mi_compute_time = mi_compute_time
        total_mi_comm_time = mi_comm_time
    
    # Training loop
    for epoch in range(config.epochs):
        # Generate training data for this epoch
        distributor.generate_subdata(config.subset_update_prob)
        
        print(f"\n[Epoch {epoch+1}/{config.epochs}]")
        
        epoch_train_time = 0.0
        epoch_comm_time = 0.0
        
        # Train
        for _, data_ptr, label in distributor.subdata:
            # Dynamic mode: Select clients before each step
            if args.mi_mode == 'dynamic':
                estimate_data = distributor.generate_estimate_subdata(config.estimate_samples)
                scores, mi_comm_time, mi_compute_time = splitnn.group_testing(estimate_data, config.n_tests)
                
                total_mi_compute_time += mi_compute_time
                total_mi_comm_time += mi_comm_time
                
                selected = [k for k, v in splitnn.selected.items() if v]
            
            label = label.to(device)
            
            # 打印维度信息（只打印一次）
            if not dim_info_printed:
                print(f"\n{'='*60}")
                print("[Data Dimensions]")
                print(f"{'='*60}")
                for i in range(config.n_clients):
                    client_id = f"client_{i}"
                    input_shape = data_ptr[client_id].shape
                    input_size = input_shape[0] * input_shape[1]
                    print(f"  {client_id} input: {tuple(input_shape)} = {input_size} elements")
                print(f"\n  [After client forward (activation to transmit)]")
                for i in range(config.n_clients):
                    client_id = f"client_{i}"
                    with torch.no_grad():
                        output = splitnn.models[client_id](data_ptr[client_id])
                        output_shape = output.shape
                        output_size = output_shape[0] * output_shape[1]
                        print(f"  {client_id} activation: {tuple(output_shape)} = {output_size} elements")
                print(f"\n  [Server receives from {len(selected)} selected clients]")
                total_activation_size = 0
                for client_id in selected:
                    with torch.no_grad():
                        output = splitnn.models[client_id](data_ptr[client_id])
                        total_activation_size += output.numel()
                print(f"  Total activation size: {total_activation_size} elements = {total_activation_size * 4 / 1024:.2f} KB")
                print(f"{'='*60}\n")
                dim_info_printed = True
            
            loss, train_time, comm_time = splitnn.train_step(data_ptr, label, config.local_epochs)
            
            global_step += 1
            epoch_train_time += train_time
            epoch_comm_time += comm_time
            total_train_time += train_time
            total_comm_time += comm_time
            
            # Periodic evaluation
            if global_step % config.eval_every_steps == 0:
                acc = evaluate(splitnn, distributor.test_set[:10], device)
                
                # Total accumulated time
                overall_total_time = (total_train_time + total_comm_time + 
                                     total_mi_compute_time + total_mi_comm_time)
                
                print(f"  Step {global_step:4d} | Selected: {selected} | Loss: {loss:.4f} | Acc: {acc*100:5.2f}%")
                
                # Dynamic mode: Show MI time for each step
                if args.mi_mode == 'dynamic':
                    print(f"         Step: {train_time + comm_time + mi_compute_time + mi_comm_time:.3f}s")
                    print(f"           - Train: {train_time:.3f}s")
                    print(f"           - Comm: {comm_time:.4f}s")
                    print(f"           - MI Compute: {mi_compute_time:.3f}s")
                    print(f"           - MI Comm: {mi_comm_time:.4f}s")
                else:
                    print(f"         Step: {train_time + comm_time:.3f}s")
                    print(f"           - Train: {train_time:.3f}s")
                    print(f"           - Comm: {comm_time:.4f}s")
                
                print(f"         Total: {overall_total_time:.2f}s")
                print(f"           - Train: {total_train_time:.2f}s")
                print(f"           - Comm: {total_comm_time:.2f}s")
                print(f"           - MI Compute: {total_mi_compute_time:.2f}s")
                print(f"           - MI Comm: {total_mi_comm_time:.2f}s")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("[Final Evaluation]")
    print(f"{'='*60}")
    
    final_acc = evaluate(splitnn, distributor.test_set, device)
    total_time = (total_train_time + total_comm_time + 
                 total_mi_compute_time + total_mi_comm_time)
    
    print(f"Accuracy: {final_acc*100:.2f}%")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  - Training Time: {total_train_time:.2f}s")
    print(f"  - Communication Time: {total_comm_time:.2f}s")
    print(f"  - MI Compute Time: {total_mi_compute_time:.2f}s")
    print(f"  - MI Comm Time: {total_mi_comm_time:.2f}s")
    print(f"Data Transferred: {comm_estimator.total_data_mb:.2f} MB")
    print(f"Encryption: {config.encryption}")
    print(f"MI Mode: {args.mi_mode}")
    print(f"Local Epochs: {config.local_epochs}")
    print(f"Total Steps: {global_step}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()