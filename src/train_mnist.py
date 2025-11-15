#!/usr/bin/env python3
"""
MNIST Digit Classifier - Your First ML Project on Pop!_OS!
A simple neural network to recognize handwritten digits (0-9)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from pathlib import Path

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

def train_model(epochs=5):
    """Train the model"""
    
    # Data preparation
    print("\n📦 Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🧠 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print(f"\n🏋️  Training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress update every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"   Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {train_loss/(batch_idx+1):.4f} | "
                      f"Acc: {100.*correct/total:.2f}%")
        
        epoch_time = time.time() - start_time
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        print(f"\n✅ Epoch {epoch+1}/{epochs} Complete!")
        print(f"   Time: {epoch_time:.2f}s")
        print(f"   Train Accuracy: {100.*correct/total:.2f}%")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print("=" * 60)
    
    # Save model
    model_path = Path('models/mnist_model.pth')
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    return model, test_acc

def main():
    print("\n" + "=" * 60)
    print("🎯 MNIST DIGIT CLASSIFIER - Your First ML Project!")
    print("=" * 60)
    
    # Train the model
    model, final_acc = train_model(epochs=5)
    
    print("\n" + "=" * 60)
    print(f"🎉 TRAINING COMPLETE!")
    print(f"   Final Test Accuracy: {final_acc:.2f}%")
    print(f"   Model saved in: models/mnist_model.pth")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
