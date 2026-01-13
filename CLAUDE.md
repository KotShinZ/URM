# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Universal Reasoning Model (URM) is a research project that achieves state-of-the-art performance on complex reasoning tasks (ARC-AGI 1: 53.8% pass@1, ARC-AGI 2: 16.0% pass@1). The project implements recurrent transformer architectures with adaptive computation time (ACT) for non-autoregressive reasoning.

**Key insight**: The improvements come from recurrent inductive bias and strong nonlinear components rather than elaborate architectural designs. URM enhances the universal transformer with short convolution (ConvSwiGLU) and truncated backpropagation.

## Environment Setup

```bash
# Install dependencies
uv venv
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation

# Setup WandB for experiment tracking
wandb login YOUR_API_KEY
```

## Data Preparation

### ARC-AGI Dataset
```bash
# ARC-AGI-1 (training + evaluation + concept)
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2 (training2 + evaluation2 + concept)
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

### Sudoku Dataset
```bash
python -m data.build_sudoku_dataset \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000
```

Data format: Processed into numpy arrays (inputs, labels, puzzle_identifiers, puzzle_indices, group_indices) with mmap support for memory efficiency.

## Training

### Main Training Script
Uses `pretrain.py` with Hydra configuration management.

```bash
# Basic training command structure
torchrun --nproc-per-node N_GPUS pretrain.py \
  data_path=DATA_PATH \
  arch=ARCH_NAME \
  [arch parameters] \
  [training hyperparameters] \
  +run_name=RUN_NAME \
  +checkpoint_path=CHECKPOINT_PATH
```

### Reproducing Paper Results
```bash
# ARC-AGI 1
bash scripts/URM_arcagi1.sh

# ARC-AGI 2
bash scripts/URM_arcagi2.sh

# Sudoku
bash scripts/URM_sudoku.sh
```

### Important Training Parameters
- `arch`: Model architecture (urm, trm, hrm)
- `arch.loops`: Number of inference steps (default: 16)
- `arch.H_cycles`: High-level cycles in recurrent processing
- `arch.L_cycles`: Low-level cycles per high-level iteration
- `arch.num_layers`: Number of transformer layers
- `global_batch_size`: Total batch size across all GPUs
- `epochs`: Training iterations through dataset
- `eval_interval`: Evaluate every N epochs
- `ema`: Enable Exponential Moving Average (recommended: True)
- `ema_rate`: EMA decay rate (default: 0.999)

### Distributed Training
Uses PyTorch's `torchrun` for multi-GPU training. The code automatically handles:
- NCCL backend for GPU communication
- GLOO backend for CPU operations
- Gradient accumulation with `grad_accum_steps`
- Per-GPU batch size = `global_batch_size / num_gpus`

## Model Architectures

### URM (Universal Reasoning Model)
**Location**: `models/urm/urm.py`
- Main architecture with best performance
- Uses ConvSwiGLU for stronger nonlinearity
- Recurrent processing with H_cycles and L_cycles
- Adaptive Computation Time (ACT) with Q-learning for dynamic halting
- Config: `config/arch/urm.yaml`

### TRM (Tiny Recursive Model)
**Location**: `models/trm/trm.py`
- Baseline recursive transformer architecture
- Option to use MLP instead of attention on sequence dimension (`mlp_t`)
- Two-level hierarchy: H (high) and L (low) states
- Config: `config/arch/trm.yaml`

### HRM (Hierarchical Reasoning Model)
**Location**: `models/hrm/`
- Alternative hierarchical architecture
- Config: `config/arch/hrm.yaml`

### Shared Components
**Location**: `models/layers.py`, `models/common.py`
- `Attention`: Multi-head self-attention with RoPE
- `ConvSwiGLU`: Convolution + SwiGLU activation (URM)
- `SwiGLU`: Gated linear unit with Swish activation (TRM)
- `RotaryEmbedding`: Rotary Position Embeddings
- `CastedSparseEmbedding`: Puzzle embeddings with sparse gradients (in `models/sparse_embedding.py`)

### Loss Functions
**Location**: `models/losses.py`
- `ACTLossHead`: Adaptive Computation Time wrapper with Q-learning
- `stablemax_cross_entropy`: Numerically stable cross-entropy loss
- Q-learning for halting: Learns when to stop recurrent processing

## Architecture Details

### Recurrent Processing
- **Carry state**: Maintains hidden states across inference steps
- **H_cycles**: Number of cycles through the model before gradient computation (only last cycle gets gradients - truncated BPTT)
- **L_cycles**: Number of input re-injections per H_cycle
- **loops**: Maximum inference steps during evaluation

### Puzzle Embeddings
Each puzzle gets a learnable sparse embedding that provides task-specific context. Updated using `CastedSparseEmbeddingSignSGD_Distributed` optimizer with separate learning rate (`puzzle_emb_lr`).

### Non-Autoregressive Design
Unlike language models, URM processes the entire sequence simultaneously and iteratively refines predictions through recurrent steps.

## Configuration System

### Hydra Configuration
- Base config: `config/cfg_pretrain.yaml`
- Architecture configs: `config/arch/{urm,trm,hrm}.yaml`
- Override with command-line: `param=value` or `+param=value` (new param)
- Nested params: `arch.loops=16`

### Key Config Sections
- `arch`: Model architecture and parameters
- `arch.loss`: Loss function configuration
- `evaluators`: List of evaluation metrics
- Hyperparameters: `lr`, `weight_decay`, `puzzle_emb_lr`, etc.

## Evaluation

### Evaluator System
**Location**: `evaluators/arc.py`

The ARC evaluator:
- Crops predictions back to original grid size
- Supports majority voting across multiple samples
- Computes pass@K metrics (K=1,2,5,10,100,1000)
- Handles puzzle identifier mapping
- Aggregates predictions across augmented versions

### Evaluation During Training
- Controlled by `eval_interval` (epochs between evaluations)
- Uses EMA weights if `ema=True`
- Can evaluate with different loop counts via `loop_deltas` (e.g., `[0, 8]` evaluates at loops and loops+8)
- Results logged to WandB

### Standalone Evaluation
Use `evaluate_trained_model.py` for evaluating saved checkpoints.

## Checkpointing

### Automatic Checkpointing
- Checkpoints saved to `checkpoint_path`
- Format: `step_N.pt` where N is training step
- Contains: model weights, optimizer states, RNG states, step count
- Config saved as `config.yaml` and `config.json`

### Loading Checkpoints
```python
# In command line
load_checkpoint=path/to/checkpoint \
load_strict=True \  # Require exact parameter match
load_optimizer_state=True  # Restore optimizer state
```

Special value: `load_checkpoint=latest` loads most recent checkpoint from `checkpoint_path`.

## Common Development Workflows

### Adding a New Model Architecture
1. Create model file in `models/your_model/your_model.py`
2. Implement class with `__init__(config_dict: dict)` and `forward()` matching the signature in existing models
3. Must implement `initial_carry()` and return a carry object with states
4. Create config in `config/arch/your_model.yaml`
5. Set `name` in config to `your_model.your_model@ClassName`

### Modifying Training Hyperparameters
- Quick test: Override via command line `lr=1e-3`
- Permanent change: Edit `config/cfg_pretrain.yaml` or architecture config
- Architecture-specific: Edit `config/arch/{arch_name}.yaml`

### Adding New Evaluation Metrics
1. Create evaluator in `evaluators/your_evaluator.py`
2. Implement `begin_eval()`, `update_batch(batch, preds)`, `result(...)`
3. Specify `required_outputs` as class attribute (tensors needed from model)
4. Add to config: `evaluators: [{name: your_module@YourClass}]`

## Data Pipeline

### Dataset Structure
**Location**: `puzzle_dataset.py`

```
data/{dataset_name}/
├── train/
│   ├── dataset.json          # Metadata
│   ├── {set}_inputs.npy      # Input sequences
│   ├── {set}_labels.npy      # Target sequences
│   ├── {set}_puzzle_identifiers.npy
│   ├── {set}_puzzle_indices.npy
│   └── {set}_group_indices.npy
└── test/
    └── (same structure)
```

### Data Augmentation
- Dihedral group (8 transformations): rotations and reflections
- Translation augmentation: Random placement in 30x30 grid
- Controlled by `num_aug` parameter during dataset building

### PuzzleDataset Class
- Lazy loads data with mmap for memory efficiency
- Supports two modes: `train` (random sampling) and `test` (sequential)
- Returns `(set_name, batch, global_batch_size)` tuples
- Handles distributed training via `rank` and `num_replicas`

## Optimizer Configuration

Uses two separate optimizers:
1. **Sparse Puzzle Embeddings**: `CastedSparseEmbeddingSignSGD_Distributed`
   - Learning rate: `puzzle_emb_lr`
   - Weight decay: `puzzle_emb_weight_decay`

2. **Model Parameters**: `AdamAtan2` (default) or `Muon` (if `use_muon=True`)
   - Learning rate: `lr` with cosine schedule
   - Warmup: `lr_warmup_steps`
   - Min ratio: `lr_min_ratio`
   - Weight decay: `weight_decay`
   - Betas: `(beta1, beta2)` default (0.9, 0.95)

## Debugging and Profiling

- Set `arch.profile=True` to enable timing information
- Disable compilation: `DISABLE_COMPILE=1 python pretrain.py ...`
- Check model parameters: Printed at start of training
- WandB tracks: loss, learning rate, custom metrics from evaluators

## Important Implementation Notes

- **Truncated BPTT**: Gradients only flow through the last H_cycle to save memory
- **Detached carry**: Carry state is detached between steps to prevent gradient accumulation
- **Q-learning for ACT**: Model learns two Q-values (halt, continue) to decide when to stop processing
- **Exploration during training**: Random early halting with probability `halt_exploration_prob` to encourage learning
- **EMA for evaluation**: Exponential moving average of weights often gives better test performance
- **Gradient accumulation**: Use `grad_accum_steps > 1` to simulate larger batch sizes
- **Target Q update**: Q-learning target updated every `target_q_update_every` steps

## Reproducing TRM Sudoku Baseline

Note: The paper's TRM architecture differs from the original TRM paper for Sudoku. This codebase uses the unified architecture:

```bash
# From original TRM repo
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 --num-aug 1000

python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  +ema=True
```
