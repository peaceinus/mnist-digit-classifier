#!/usr/bin/env python3
"""
Main script for the project
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def main():
    print("Project initialized!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
if __name__ == "__main__":
    main()
