# Federated Learning for Disposable Income Regression

## Project Overview

This project implements advanced Federated Learning strategies for disposable income prediction using the Flower framework with the Indian Personal Finance dataset. It features multiple FL strategies (FedAvg, FedProx, SCAFFOLD, and an enhanced Hybrid approach), sophisticated data partitioning methods, real-time visualization, and comprehensive experiment management through Hydra configuration.

## Recent Changes (January 2026)

### 🚀 Enhanced Hybrid FL Strategy
- **Adaptive Weight Balancing**: Progressive transition from FedProx-dominated (early rounds) to SCAFFOLD-dominated (later rounds)
  - Early rounds (0-10): FedProx weight 0.9, SCAFFOLD weight 0.2
  - Later rounds (20+): FedProx weight 0.6, SCAFFOLD weight 0.7
- **Momentum-Enhanced Control Variates**: 90% momentum on SCAFFOLD corrections for smoother convergence
- **Deeper Neural Network**: Enhanced from 128→64 to 160→96→48 architecture (20K+ parameters)
- **Adaptive Learning Rate Strategy**: Dynamic LR adjustment with 50% boost in early rounds
- **Dynamic Mu Adjustment**: Progressive reduction from 0.08 to 0.03 through training
- **Expected Performance**: 15-30% RMSE reduction over FedAvg, R² of 0.85-0.90

### 📊 Data Partitioning Strategies
- **Hybrid Partitioning**: 12 clients based on City_Tier × Occupation combinations
  - Maximum heterogeneity for realistic FL scenarios
  - Varying client sizes: Tier_2 (~2000 samples), Tier_1 (~1200), Tier_3 (~800)
- **Dirichlet Partitioning**: Configurable data heterogeneity via alpha parameter
  - Alpha 0.1: Extreme non-IID (50-100x sample imbalance)
  - Alpha 0.5: High non-IID (recommended for testing)
  - Alpha 1.0: Moderate non-IID (typical FL)
  - Alpha 10.0+: Near IID distribution
- **City Tier**: 3 clients (Tier_1, Tier_2, Tier_3)
- **Occupation**: 4 clients (Retired, Professional, Student, Self_Employed)

### 🎯 Model & Training Optimizations
- **Log-Scale Target Transformation**: 30-50% MAPE reduction via y' = log(1+y)
- **Feature Engineering**: 6 new features (Total_Expenses, Expense_to_Income_Ratio, Essential/Discretionary_Expenses, Age², log(Income))
- **AdamW Optimizer**: Adaptive learning with decoupled weight decay (1e-4)
- **Cosine Annealing LR Scheduler**: Smooth learning rate decay
- **Gradient Clipping**: max_norm=1.0 for training stability
- **LayerNorm**: FL-safe normalization without global batch statistics dependency
- **GELU Activation**: Smoother gradients vs ReLU, more robust to client drift

### 📈 Visualization & Analysis
- **Real-time Metrics Plotting**: Live visualization during training
- **Comprehensive Comparison**: Automatic generation of strategy comparison plots
- **CSV Exports**: Detailed metrics history for further analysis
- **Organized Outputs**: Hydra-managed experiment directories with timestamps

## Project Structure

```
regression/
├── README.md                           # This file
└── flowertry/                          # Main project directory
    ├── main.py                         # Main entry point with Hydra
    ├── client.py                       # FL client implementation
    ├── server.py                       # FL server strategies
    ├── model.py                        # Neural network (160→96→48)
    ├── dataset.py                      # Dataset & partitioning
    ├── visualize_metrics.py            # Real-time visualization
    ├── compare_strategies.py           # Strategy comparison
    ├── plot_metrics_progression.py     # Metrics plotting
    ├── analyze_partitioning.py         # Data analysis
    │
    ├── conf/                           # Configuration files
    │   ├── base.yaml                   # Main configuration
    │   ├── improved_strategies.yaml    # Strategy configs
    │   └── optimized.yaml              # Hyperparameters
    │
    ├── data/                           # Datasets
    │   └── IndianPersoalFinance/
    │       └── indianPersonalFinanceAndSpendingHabits.csv
    │
    ├── outputs/                        # Results (timestamped)
    │   └── YYYY-MM-DD/HH-MM-SS/
    │
    └── Documentation/                  # Guides
        ├── README.md                   # Full documentation
        ├── QUICK_START.md              # Quick guide
        ├── HYBRID_IMPROVEMENTS.md      # Technical details
        ├── HYBRID_PARTITIONING.md      # Partitioning guide
        ├── DIRICHLET_PARTITIONING.md   # Dirichlet guide
        └── STRATEGY_IMPROVEMENTS.md    # Optimizations
```

## Installation

```bash
# Create environment
conda create -n flower_fl python=3.10
conda activate flower_fl

# Install dependencies
pip install flwr==1.20.0 torch torchvision numpy pandas scikit-learn matplotlib hydra-core omegaconf
```

## Quick Start

```bash
cd flowertry

# Run default (FedAvg)
python main.py

# Run specific strategy
python main.py strategy=hybrid
python main.py strategy=fedprox
python main.py strategy=scaffold

# Compare all strategies
python main.py compare_all=true
```

## Data Partitioning Examples

```bash
# Hybrid partitioning (12 clients: City_Tier × Occupation)
python main.py partition_strategy=hybrid num_clients=12

# Dirichlet (high heterogeneity)
python main.py partition_strategy=dirichlet dirichlet_alpha=0.5

# City tier (3 clients)
python main.py partition_strategy=city_tier num_clients=3

# Occupation (4 clients)
python main.py partition_strategy=occupation num_clients=4
```

## Key Features

### FL Strategies

| Strategy   | Description                              | RMSE          | R²            |
| ---------- | ---------------------------------------- | ------------- | ------------- |
| FedAvg     | Baseline federated averaging             | 3500-4000     | 0.75-0.80     |
| FedProx    | Proximal term regularization             | 3000-3500     | 0.80-0.85     |
| SCAFFOLD   | Control variate variance reduction       | 3200-3800     | 0.78-0.83     |
| **Hybrid** | **Adaptive FedProx + SCAFFOLD**          | **2500-3000** | **0.85-0.90** |

### Model Architecture

```
DisposableIncomeNet:
  Input: 25 features
  Hidden: 160 → 96 → 48
  Output: 1 (disposable income)
  Activation: GELU
  Normalization: LayerNorm
  Dropout: 0.18
```

### Configuration

Main parameters in `conf/base.yaml`:

```yaml
strategy: fedavg
num_rounds: 25
num_clients: 12
local_epochs: 3
batch_size: 32
learning_rate: 0.0012
partition_strategy: hybrid
```

## Documentation

Detailed guides in `flowertry/` directory:

- **[README.md](flowertry/README.md)**: Complete documentation
- **[QUICK_START.md](flowertry/QUICK_START.md)**: 5-minute guide
- **[HYBRID_IMPROVEMENTS.md](flowertry/HYBRID_IMPROVEMENTS.md)**: Technical details
- **[HYBRID_PARTITIONING.md](flowertry/HYBRID_PARTITIONING.md)**: Partitioning guide
- **[DIRICHLET_PARTITIONING.md](flowertry/DIRICHLET_PARTITIONING.md)**: Dirichlet guide
- **[STRATEGY_IMPROVEMENTS.md](flowertry/STRATEGY_IMPROVEMENTS.md)**: Optimizations
- **[STRATEGY_CONFIGS.md](flowertry/STRATEGY_CONFIGS.md)**: Config reference

## Testing

```bash
cd flowertry

# Test hybrid strategy
./test_hybrid.sh

# Test improvements
python test_improvements.py

# Test partitioning
python test_hybrid_partitioning.py
python test_dirichlet.py
```

## Output Files

Each experiment creates:
- `metrics_history.csv`: Training metrics per round
- `comparison_results.json`: Strategy comparison data
- Visualization plots (PNG): RMSE, MAE, R², MAPE
- Hydra configuration logs

## Environment

- Python 3.10
- Flower 1.20.0
- PyTorch, torchvision
- NumPy, Pandas, scikit-learn
- Matplotlib, Hydra, OmegaConf
- Optimized for macOS M1 Pro

## Advanced Usage

### Hyperparameter Tuning

```bash
# Tune hybrid weights
python main.py strategy=hybrid hybrid.fedprox_weight=0.3 hybrid.scaffold_weight=0.5

# Adjust FedProx mu
python main.py strategy=fedprox fedprox.mu=0.2

# Extended training
python main.py num_rounds=100 local_epochs=8
```

### Multiple Experiments

```bash
# Test different alpha values
for alpha in 0.1 0.5 1.0 5.0 10.0; do
  python main.py partition_strategy=dirichlet dirichlet_alpha=$alpha num_rounds=25
done
```

## Performance Tips

1. **Use Hybrid partitioning** for realistic non-IID testing
2. **Run 45+ rounds** for Hybrid strategy to show full benefits
3. **Adjust learning rate** if convergence is too slow/fast
4. **Monitor outputs/** directory for results and visualizations

## References

- [Flower Framework](https://flower.dev/docs/)
- [FedProx Paper](https://arxiv.org/abs/1812.06127)
- [SCAFFOLD Paper](https://arxiv.org/abs/1910.06378)

---

**Last Updated**: January 18, 2026  
**Status**: Active Development  
**Dataset**: Indian Personal Finance (Kaggle)
