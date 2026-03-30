# 使用包含 nvcc 编译器的 devel 镜像，防止 EVSSM 等算法在容器内找不到 CUDA 编译器报错
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PATH=/opt/conda/bin:$PATH

# 安装必要的系统库
RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -f /tmp/miniconda.sh

# 设置工作目录
WORKDIR /root/IR-service

# 关键：先只拷贝环境配置文件和脚本，利用 Docker 缓存加速后续修改
COPY envirnoment/ ./envirnoment/
COPY docker/ ./docker/

# 运行刚才写的批量创建环境脚本
RUN bash docker/create_envs.sh

# 最后拷贝你的所有源代码和模型权重
COPY . /root/IR-service

EXPOSE 6006

CMD ["bash", "docker/start.sh"]
