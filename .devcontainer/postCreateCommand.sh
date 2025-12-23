#!/bin/bash

# GitHub SSH設定スクリプト
# コンテナ作成後に自動実行される

echo "Setting up GitHub SSH configuration..."

# SSHディレクトリが存在しない場合は作成
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# SSH設定ファイルを作成/更新
cat > ~/.ssh/config << 'EOF'
Host github.com
  HostName ssh.github.com
  User git
  Port 443
  StrictHostKeyChecking accept-new
EOF
  #ProxyCommand connect -H http://proxy11.omu.ac.jp:8080 %h %p

chmod 600 ~/.ssh/config

#git config --global user.name "Your Name"
#git config --global user.email "your.email@example.com"

echo "GitHub SSH setup completed!"

[ -d ".venv" ] || uv venv
source .venv/bin/activate
uv sync --active

# uv pip install ./reversible-deq
#pip install -r requirements.txt

# git config --global user.name "Your Name"
# git config --global user.email "your.email@example.com"

# ./.devcontainer/sync_claude.sh

# JAXのモデル特有の警告を抑制
#export XLA_FLAGS=--xla_gpu_enable_analytical_sol_latency_estimator=false

# 初めにすること
# git config --global user.name "Your Name"
# git config --global user.email "your.email@example.com"
# cd nanochat
# uv venv
# source .venv/bin/activate
# uv sync --active --extra gpu
# hf auth login
# wandb login
# bash speedrun.sh

