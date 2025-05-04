# This script was created following the official documentation
# for a Ubuntu Worsktation 20.04

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install nvidia-open
apt install cuda-drivers