#!/usr/bin/env bash
set -e

# 激活 conda 基础命令
source /opt/conda/etc/profile.d/conda.sh
cd /root/IR-service

echo "开始读取 envirnoment 文件夹中的 .yml 文件..."

# 遍历文件夹下的所有 yml 文件并自动创建环境
for yml_file in envirnoment/*.yml; do
    if [ -f "$yml_file" ]; then
        echo "========================================"
        echo "正在从 $yml_file 恢复环境..."
        echo "========================================"
        # 使用 conda env create 直接从 yml 还原
        conda env create -f "$yml_file"
    fi
done

echo "所有 Conda 环境创建完毕，当前环境列表："
conda env list

# 清理 conda 缓存，大幅减小最终 Docker 镜像的体积
conda clean --all -y
