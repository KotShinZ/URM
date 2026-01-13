run_name="URM-sudoku2"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
data_path=data/sudoku-extreme-1k-aug-1000 \
arch=urm \
evaluators="[]" \
epochs=50000 \
eval_interval=2000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
