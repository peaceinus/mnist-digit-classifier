# 🎯 MNIST Digit Classifier

**My First ML Project on Pop!_OS!**

## 📝 Description
A neural network that recognizes handwritten digits (0-9) using the MNIST dataset.

## 🏗️ Architecture
- Input: 28x28 grayscale images
- Layer 1: 784 → 128 (ReLU + Dropout)
- Layer 2: 128 → 64 (ReLU + Dropout)
- Output: 64 → 10 (digit classes)

## 🚀 Quick Start
```bash
# Activate environment
conda activate ml-env

# Train model
python src/train_mnist.py

# Visualize predictions
python src/visualize_results.py
```

## 📈 Results
- **Training Accuracy:** ~98%
- **Test Accuracy:** ~97%
- **Training Time:** ~2-3 minutes on GTX 1650
- **Model Size:** ~100K parameters

## 🛠️ Tech Stack
- Python 3.11
- PyTorch 2.5+ (CUDA enabled)
- NVIDIA GTX 1650 GPU
- Pop!_OS Linux

## 📁 Project Structure
```
mnist-digit-classifier/
├── data/                  # MNIST dataset (auto-downloaded)
├── models/                # Saved model weights
│   └── mnist_model.pth
├── results/               # Output visualizations
│   └── predictions.png
├── src/                   # Source code
│   ├── train_mnist.py    # Training script
│   └── visualize_results.py
├── notebooks/             # Jupyter notebooks
└── README.md
```

## 🎓 What I Learned
1. Setting up ML development environment on Linux
2. Using PyTorch for deep learning
3. Training neural networks on GPU
4. Evaluating model performance
5. COSMIC desktop workflow for ML development

## 🔥 Next Steps
- [ ] Experiment with different architectures
- [ ] Add data augmentation
- [ ] Try different optimizers
- [ ] Deploy model as web app
- [ ] Move to more complex datasets (CIFAR-10)

---

**Built with ❤️ on Pop!_OS COSMIC**
