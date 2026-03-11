FROM nvidia/cuda:12.8.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAMBA_ROOT_PREFIX=/opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

RUN micromamba create -y -n app python=3.10 pip && micromamba clean --all --yes

WORKDIR /app

COPY requirements.txt ./
RUN micromamba run -n app pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["micromamba", "run", "-n", "app", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]