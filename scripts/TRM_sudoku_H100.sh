run_name="pretrain_att_sudoku"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node 1 pretrain.py \
arch=trm \
data_path=data/sudoku-extreme-1k-aug-1000 \
evaluators="[]" \
epochs=50000 eval_interval=2000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} \
+checkpoint_path=${checkpoint_path} \
+ema=True
