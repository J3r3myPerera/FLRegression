"""Simple smoke test to verify environment"""
import sys
print("Testing imports...")

try:
    import numpy as np
    print("✓ numpy")
    import pandas as pd
    print("✓ pandas")
    import torch
    print("✓ torch")
    import flwr as fl
    print("✓ flwr")
    import hydra
    print("✓ hydra")
    import matplotlib
    print("✓ matplotlib")
    print("\nAll imports successful!")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
