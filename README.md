# Universal Reasoning Model

[![paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.14693)

Universal transformers (UTs) have been widely used for complex reasoning tasks such as ARC-AGI and Sudoku, yet the specific sources of their performance gains remain underexplored. In this work, we systematically analyze UTs variants and show that improvements on ARC-AGI primarily arise from the recurrent inductive bias and strong nonlinear components of Transformer, rather than from elaborate architectural designs. Motivated by this finding, we propose the Universal Reasoning Model (URM), which enhances the UT with short convolution and truncated backpropagation. Our approach substantially improves reasoning performance, achieving state-of-the-art 53.8% pass@1 on ARC-AGI 1 and 16.0% pass@1 on ARC-AGI 2.


## Installation
```bash
uv venv
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation 
```

## Login Wandb
```bash
wandb login YOUR_API_KEY
```

## Preparing Data
```bash
# ARC-AGI-1
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Sudoku
python -m data.build_sudoku_dataset \
  --output-dir data/sudoku-extreme-1k-aug-1000  \
  --subsample-size 1000 \
  --num-aug 1000

# upload ARC-AGI-1
export HF_TOKEN=YOUR_HF_TOKEN
python -m data.upload_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --subsets training evaluation \
  --test-set-name evaluation \
  --hf-repo-id "your-username/arc-agi-augmented" \
  --hf-token $HF_TOKEN \
  --num-aug 100

# upload ARC-AGI-2
export HF_TOKEN=YOUR_HF_TOKEN
python -m data.upload_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --subsets training2 evaluation2 \
  --test-set-name evaluation2 \
  --hf-repo-id "your-username/arc-agi-augmented" \
  --hf-token $HF_TOKEN
```

## Reproducing ARC-AGI 1 Score
```bash
bash scripts/URM_arcagi1.sh
```

## Reproducing ARC-AGI 2 Score
```bash
bash scripts/URM_arcagi2.sh
```

## Reproducing Sudoku Score
```bash
bash scripts/URM_sudoku.sh
```

### Citation
```
@misc{gao2025universalreasoningmodel,
      title={Universal Reasoning Model}, 
      author={Zitian Gao and Lynx Chen and Yihao Xiao and He Xing and Ran Tao and Haoming Luo and Joey Zhou and Bryan Dai},
      year={2025},
      eprint={2512.14693},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.14693}, 
}
```
