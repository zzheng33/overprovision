"""
CNN Training with Multi-GPU Support
Supports multiple CNN architectures: LeNet, ResNet, VGG
"""

# Core PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Torchvision for datasets and transforms
from torchvision import datasets, transforms

# Standard libraries
import numpy as np
import time
import os
import sys
from typing import Tuple, List, Optional
import argparse

# Import models from models folder
from models import ResNet50, VGG16

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# Model registry
MODEL_REGISTRY = {
    'resnet50': ResNet50,
    'vgg16': VGG16,
}


def setup_device(num_gpus=None):
    """
    Setup device configuration for training

    Args:
        num_gpus: Number of GPUs to use. If None, uses all available GPUs.
                 If 0, uses CPU. If > 0, uses that many GPUs.

    Returns:
        device: torch.device object
        gpu_ids: list of GPU IDs to use
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu'), []

    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    if num_gpus is None:
        num_gpus = available_gpus
    elif num_gpus == 0:
        print("Using CPU (num_gpus=0)")
        return torch.device('cpu'), []
    elif num_gpus > available_gpus:
        print(f"Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        num_gpus = available_gpus

    gpu_ids = list(range(num_gpus))
    device = torch.device(f'cuda:{gpu_ids[0]}')
    print(f"Using {num_gpus} GPU(s): {gpu_ids}")

    return device, gpu_ids


def create_model(model_name='lenet', num_gpus=None, num_classes=10, input_channels=1):
    """
    Create model with multi-GPU support

    Args:
        model_name: Name of the model architecture
        num_gpus: Number of GPUs to use
        num_classes: Number of output classes
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)

    Returns:
        model: model instance
        device: device being used
    """
    device, gpu_ids = setup_device(num_gpus)

    # Get model class from registry
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    model = model_class(num_classes=num_classes, input_channels=input_channels)
    print(f"Using {model_name.upper()} model")

    model = model.to(device)

    if len(gpu_ids) > 1:
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    return model, device


def train(model, device, train_loader, optimizer, criterion, epoch):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_start_time = time.time()
    processed_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Track processed samples and calculate running throughput
        processed_samples += len(data)
        if batch_idx % 100 == 0 and batch_idx > 0:
            elapsed_time = time.time() - epoch_start_time
            running_throughput = processed_samples / elapsed_time
            print(f'Epoch {epoch} [Step {batch_idx}/{len(train_loader)}] Throughput: {running_throughput:.2f} images/sec')

    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_time = time.time() - epoch_start_time
    epoch_throughput = processed_samples / epoch_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    print(f'Epoch {epoch} Complete - Throughput: {epoch_throughput:.2f} images/sec')
    return epoch_loss, epoch_acc, epoch_time


def test(model, device, test_loader, criterion):
    """Evaluate the model on test set"""
    model.eval()
    test_loss = 0
    correct = 0

    test_start_time = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_time = time.time() - test_start_time
    test_throughput = len(test_loader.dataset) / test_time
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    # print(f'Test - Throughput: {test_throughput:.2f} images/sec\n')
    print()
    return test_loss, accuracy


def prepare_data(batch_size=64):
    """
    Prepare MNIST dataset for training

    Args:
        batch_size: Batch size for data loaders

    Returns:
        train_loader, test_loader: DataLoader objects
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def print_system_info():
    """Print PyTorch and CUDA system information"""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 60)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='CNN Training with Multi-GPU Support')
    parser.add_argument('--model', type=str, default='lenet',
                        choices=['lenet', 'resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg16'],
                        help='Model architecture (default: lenet)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1, 0 for CPU)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Save the trained model')
    parser.add_argument('--model-path', type=str, default='model.pth',
                        help='Path to save the model (default: model.pth)')

    args = parser.parse_args()

    # Print system information
    # print_system_info()

    # Configuration
    NUM_GPUS = args.num_gpus
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    print(f"\nTraining Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Number of GPUs: {NUM_GPUS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    # Prepare data
    print("Preparing data...")
    train_loader, test_loader = prepare_data(BATCH_SIZE)

    # Create model
    print("\nCreating model...")
    model, device = create_model(model_name=args.model, num_gpus=NUM_GPUS,
                                 num_classes=10, input_channels=1)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    

    # Training loop
    print("\nStarting training...")
    start_time = time.time()

    train_losses = []
    train_accs = []
    train_times = []
    epoch_wall_times = []
    test_losses = []
    test_accs = []

    for epoch in range(1, EPOCHS + 1):
        epoch_wall_start = time.time()
        train_loss, train_acc, train_time = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        epoch_wall_time = time.time() - epoch_wall_start

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_times.append(train_time)
        epoch_wall_times.append(epoch_wall_time)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    end_time = time.time()
    total_wall_time = end_time - start_time
    if EPOCHS > 1:
        effective_epochs = EPOCHS - 1
        # Exclude epoch 1 warmup effects from summary metrics.
        total_training_time_excl_epoch1 = sum(epoch_wall_times[1:])
        total_train_time_excl_epoch1 = sum(train_times[1:])
        avg_train_time = total_train_time_excl_epoch1 / effective_epochs
        avg_train_throughput = len(train_loader.dataset) / avg_train_time
    else:
        effective_epochs = 0
        total_training_time_excl_epoch1 = 0.0
        avg_train_time = 0.0
        avg_train_throughput = 0.0

    print(f"\nTotal training time (all epochs): {total_wall_time:.2f} seconds")
    print(f"Total training time (excluding epoch 1): {total_training_time_excl_epoch1:.2f} seconds")
    if effective_epochs > 0:
        print(f"Average train time per epoch (excluding epoch 1): {avg_train_time:.2f} seconds")
        print(f"Average train throughput (excluding epoch 1): {avg_train_throughput:.2f} images/sec")
    else:
        print("Average train metrics excluding epoch 1: N/A (need at least 2 epochs)")

    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), args.model_path)
        print(f"\nModel saved to {args.model_path}")

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
