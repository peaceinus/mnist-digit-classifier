#!/usr/bin/env python3
"""
Visualize MNIST predictions
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Same model architecture
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

def visualize_predictions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load('models/mnist_model.pth'))
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )
    
    # Get random samples
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('MNIST Predictions - Your First ML Model!', fontsize=16, fontweight='bold')
    
    indices = np.random.choice(len(test_dataset), 10, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        img, label = test_dataset[indices[idx]]
        
        # Make prediction
        with torch.no_grad():
            img_gpu = img.unsqueeze(0).to(device)
            output = model(img_gpu)
            prediction = output.argmax(dim=1).item()
        
        # Display
        ax.imshow(img.squeeze(), cmap='gray')
        color = 'green' if prediction == label else 'red'
        ax.set_title(f'Pred: {prediction} | True: {label}', color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('results/predictions.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    # Also show it
    plt.show()

if __name__ == "__main__":
    visualize_predictions()
