#!/bin/bash

NGC_API_KEY="$1"

if [ -z $1 ]; then
    echo ""
    echo "ERROR. CORRECT USAGE: ./deploy.sh {NGC_API_KEY}"
    echo ""
    exit 0
fi

### Install Prerequisites ###
sudo dnf install -y tar bzip2 make automake gcc gcc-c++ git vim cairo cairo-devel libcurl-devel pciutils elfutils-libelf-devel libglvnd-devel iptables platform-python-devel.x86_64 openssl-devel dbus-devel

### Install Docker ###
sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.4.3-3.1.el7.x86_64.rpm
sudo dnf install docker-ce -y
sudo systemctl --now enable docker


### Install Container Toolkit ###
curl -s -L https://nvidia.github.io/libnvidia-container/centos7/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf clean expire-cache --refresh
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

### Add Local NVMe SSD ###
sfdisk /dev/nvme0n1 <<EOF
;
EOF
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir /data && sudo mkdir /data/docker-images
sudo mount /dev/nvme0n1 /data
sudo chown -R opc:opc /data

### Change Docker Storage Directory ###
sudo systemctl stop docker
sudo sed -i 's|dockerd -H|dockerd --data-root /data/docker -H|' /lib/systemd/system/docker.service
sudo rsync -aqxP /var/lib/docker/ /data/docker
sudo systemctl daemon-reload
sudo systemctl start docker

### Download Hugging Face Model ###
mkdir /data/models && cd /data/models
wget https://huggingface.co/nvidia/nemo-megatron-gpt-5B/resolve/main/nemo_gpt5B_bf16_tp2.nemo

### Pull NeMo-Megatron Training + Inference Containers ###
sudo docker login nvcr.io --username='$oauthtoken' --password=${NGC_API_KEY}
sudo docker pull nvcr.io/ea-bignlp/bignlp-training:22.11-py3
sudo docker pull nvcr.io/ea-bignlp/bignlp-inference:22.08-py3

### Convert Model to FasterTransformer ###
mkdir /data/models/modelsFT
modelpath=/data/models
num_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
sudo docker run --rm \
    --gpus all \
    --shm-size=16GB \
    -v $modelpath:/checkpoints \
    -v $modelpath/modelsFT:/model_repository \
    -e num_gpu=$num_gpu \
    nvcr.io/ea-bignlp/bignlp-training:22.11-py3 \
    bash -c 'export PYTHONPATH=/opt/bignlp/FasterTransformer:${PYTHONPATH} && \
    wget https://raw.githubusercontent.com/triton-inference-server/fastertransformer_backend/main/all_models/gpt/fastertransformer/config.pbtxt && \
    cd /opt/bignlp && \
    python3 FasterTransformer/examples/pytorch/gpt/utils/nemo_ckpt_convert.py \
        --in-file /checkpoints/nemo_gpt5B_bf16_tp2.nemo \
        --infer-gpu-num ${num_gpu} \
        --saved-dir /model_repository/gpt3_5b \
        --weight-data-type fp16 \
        --load-checkpoints-to-cpu 0 && \
    python3 /opt/bignlp/bignlp-scripts/bignlp/collections/export_scripts/prepare_triton_model_config.py \
        --model-train-name gpt3_5b \
        --template-path /opt/bignlp/fastertransformer_backend/all_models/gpt/fastertransformer/config.pbtxt \
        --ft-checkpoint /model_repository/gpt3_5b/${num_gpu}-gpu \
        --config-path /model_repository/gpt3_5b/config.pbtxt \
        --max-batch-size 256 \
        --pipeline-model-parallel-size 1 \
        --tensor-model-parallel-size ${num_gpu} \
        --data-type fp16'

### Deploy FT Model ###
sudo chown -R opc:opc /data/models/modelsFT/gpt3_5b
mkdir /data/models/modelsFT/gpt3_5b/1
mv /data/models/modelsFT/gpt3_5b/8-gpu /data/models/modelsFT/gpt3_5b/1
cp /data/models/modelsFT/gpt3_5b/1/8-gpu/config.ini /data/models/modelsFT/gpt3_5b/1
sed -i 's|gpt3_5b/8-gpu|gpt3_5b/1/8-gpu|' /data/models/modelsFT/gpt3_5b/config.pbtxt
sudo docker run --rm \
    --name triton-inference-server \
    -d \
    --gpus all \
    -p 8000-8002:8000-8002 \
    -v $modelpath/modelsFT:/model_repository \
    nvcr.io/ea-bignlp/bignlp-inference:22.08-py3 \
    bash -c 'tritonserver --model-repository /model_repository'

### Setup Python Virtual Environment and Install Packages ###
sudo mkdir /data/venv
sudo chown -R opc:opc /data/venv
sudo python -m venv /data/venv
source /data/venv/bin/activate
export PYCURL_SSL_LIBRARY=openssl
sudo python -m pip install --upgrade pip
sudo python -m pip install -r requirements.txt
sudo python -m pip install tritonclient[http] transformers

### Wait for Triton Inference Server to Start ###
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' http://localhost:8000/v2/health/ready)" != "200" ]]; do sleep 5; done

sudo python3 infer.py "1 2 3 4 5 6"
