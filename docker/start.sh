#!/usr/bin/env bash
set -e

source /opt/conda/etc/profile.d/conda.sh
cd /root/IR-service

# 确保运行时需要的输入输出目录存在
mkdir -p workdirs/tmp workdirs/inputs workdirs/outputs

# 启动你的 FastAPI 主服务 (请确保 irservice 环境已经存在)
echo "启动 FastAPI 主服务..."
exec conda run --no-capture-output -n irservice uvicorn main:app --host 0.0.0.0 --port 6006
