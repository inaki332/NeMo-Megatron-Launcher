# Deploy NeMo-Megatron on OCI Bare Metal Instance

## Instance Creation
1. Navigate to "Instances" on OCI dashboard.
2. Select the "Create Instance" button.
3. In "Image and shape" modify the following.
    1. Shape
        1. Select "Change Shape" button.
        2. Select "Bare metal machine" button.
        3. Select "BM.GPU.B4.8" or "BM.GPU4.8" node type.
        4. Accept the "Oracle and NVIDIA Terms of Use" and confirm shape.
    2. Image
        1. Select "Change Image" button.
        2. Toggle the &darr; on the right side of "Oracle Linux 8" and select the latest Gen2-GPU build from the "Image build" dropdown menu.
        3. Confirm image.
4. The default settings of creating a new Virtual Cloud Network (VCN) will be used.
5. In "Add SSH keys" select your preferred method of SSH authentication. **THIS IS A REQUIRED STEP**.
    1. The recommended method is to use the "Paste public keys" option and paste in a public SSH key from the machine with which you intend to access the compute instance.
6. Create the instance at the bottom of the page.

## Accessing the Instance
1. After instance creation above, select the instance in the Compute Instances menu.
2. After 2-5 minutes, the instance will be ready to SSH using the public IP address located under the "Instance access" section on the Instance details page and the SSH key pair specified during instance creation.
3. SSH into the instance using the key. The below example is when using a private key pair downloaded from OCI.
```
  chmod 600 private_key.key
  ssh -i private_key.key opc@ip_address
```

## NeMo-Megatron Deployment
Note: The NeMo-Megatron Deployment script [deploy.sh](https://github.com/inaki332/NeMo-Megatron-Launcher/blob/master/csp_tools/oci/deploy-docker-oci/deploy.sh) performs the following:
- Installation
    - [Prerequisite Packages](https://github.com/inaki332/NeMo-Megatron-Launcher/blob/master/csp_tools/oci/deploy-docker-oci/deploy.sh#:~:text=sudo%20dnf%20install%20%2Dy%20tar%20bzip2%20make%20automake%20gcc%20gcc%2Dc%2B%2B%20git%20vim%20cairo%20cairo%2Ddevel%20libcurl%2Ddevel%20pciutils%20elfutils%2Dlibelf%2Ddevel%20libglvnd%2Ddevel%20iptables%20platform%2Dpython%2Ddevel.x86_64%20openssl%2Ddevel%20dbus%2Ddevel)
    - Docker
    - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    - [Pip Requirements](https://github.com/inaki332/NeMo-Megatron-Launcher/blob/master/csp_tools/oci/deploy-docker-oci/requirements.txt)
- Downloads
    - Hugging Face [nemo_gpt5B_bf16_tp2.nemo](https://huggingface.co/nvidia/nemo-megatron-gpt-5B) Model
    - NeMo-Megatron [Training:22.11-py3](https://registry.ngc.nvidia.com/orgs/ea-bignlp/containers/bignlp-training) + [Inference:22.08-py3](https://registry.ngc.nvidia.com/orgs/ea-bignlp/containers/bignlp-inference) Containers
- Configuration
    - Partitions, formats, and mounts nvme0n1 local node storage
    - Converts Hugging Face pytorch model to [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
    - Setup of python virtual environments for package installation
- Deployment
    - Deploys Triton Inference Server using the NeMo-Megatron Inference container
    - Runs a sample [infer.py](https://github.com/inaki332/NeMo-Megatron-Launcher/blob/master/csp_tools/oci/deploy-docker-oci/infer.py) program

#### Deployment Instructions
1. Clone this repo, which contains the deployment and dependency scripts.
```
  git clone https://github.com/inaki332/NeMo-Megatron-Launcher.git
```
2. Navigate to the OCI deployment directory.
```
  cd NeMo-Megatron-Launcher/csp_tools/oci/deploy-docker-oci/
```
3. Execute the [deploy.sh](https://github.com/inaki332/NeMo-Megatron-Launcher/blob/master/csp_tools/oci/deploy-docker-oci/deploy.sh) script.
```
  ./deploy.sh
```
4. After the script's completion, a successful output from the Triton Inference Server endpoint is the below.
```
  “1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36"
```
