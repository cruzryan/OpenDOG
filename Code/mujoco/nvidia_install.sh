# This script was created following the official documentation
# for a Ubuntu Worsktation 24

apt install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/nvidia-driver/$570.144/local_installers/nvidia-driver-local-repo-ubuntu-570.144_$arch_ext.deb
dpkg -i nvidia-driver-local-repo-ubuntu-570.144_$arch_ext.deb
apt update
cp /var/nvidia-driver-local-repo-ubuntu-570.144/nvidia-driver-*-keyring.gpg /usr/share/keyrings/
apt install nvidia-open
apt install cuda-drivers