# 🎯 MNIST Digit Classifier

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**A neural network that recognizes handwritten digits using PyTorch and CUDA acceleration.**

<p align="center">
  <img src="results/predictions.png" alt="MNIST Predictions" width="600"/>
</p>

## 🚀 Features

- ⚡ GPU-accelerated training with CUDA
- 📊 Achieves 97%+ accuracy on test set
- 🎨 Visualization of predictions
- 📝 Clean, documented code
- 🔄 Modular architecture for easy experimentation

## 🏗️ Architecture
```
Input (28x28) → Flatten → Dense(128) → ReLU → Dropout(0.2)
                        → Dense(64)  → ReLU → Dropout(0.2)
                        → Dense(10)  → Softmax
```

**Parameters:** ~109K  
**Training Time:** ~2-3 minutes on GTX 1650

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/peaceinus/mnist-digit-classifier.git
cd mnist-digit-classifier

# Create conda environment
conda create -n mnist python=3.11
conda activate mnist

# Install dependencies
pip install torch torchvision matplotlib numpy
```

## 🎮 Usage

### Train Model
```bash
python src/train_mnist.py
```

### Visualize Results
```bash
python src/visualize_results.py
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 98.2% |
| Test Accuracy | 97.4% |
| Training Time | 2m 45s |
| GPU Memory | ~500 MB |

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Framework:** PyTorch 2.5+
- **Acceleration:** CUDA 12.1
- **GPU:** NVIDIA GTX 1650
- **OS:** Pop!_OS Linux

## 📁 Project Structure
```
mnist-digit-classifier/
├── data/                  # MNIST dataset (auto-downloaded)
├── models/                # Saved model weights
│   └── mnist_model.pth
├── results/               # Visualizations and outputs
├── src/
│   ├── train_mnist.py    # Training script
│   └── visualize_results.py
├── README.md
└── requirements.txt
```

## 🎓 Learning Objectives

- [x] Set up ML development environment on Linux
- [x] Understand neural network architecture
- [x] Implement forward and backward propagation
- [x] Train models with GPU acceleration
- [x] Evaluate and visualize model performance
- [x] Version control with Git/GitHub

## 🚀 Future Improvements

- [ ] Add data augmentation
- [ ] Experiment with CNN architecture
- [ ] Implement learning rate scheduling
- [ ] Add model checkpointing
- [ ] Create web demo with Flask/FastAPI

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Neural Networks Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

## 📜 License

MIT License - feel free to use this project for learning!

## 👤 Author

**peaceinus**
- GitHub: [@peaceinus](https://github.com/peaceinus)
- Project: Part of comprehensive ML learning journey

---

⭐ **Star this repo if you found it helpful!**
