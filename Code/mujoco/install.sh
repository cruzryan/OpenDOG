#!/bin/bash

#Set up system for runpod

sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl
cd /tmp
wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz

tar -xf Python-3.12.7.tgz
cd Python-3.12.7
./configure --enable-optimizations
make -j4
make altinstall
python3.12 --version
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12


# Script to install specific versions of packages with pip

# Define the packages and versions
packages=(
    "mujoco==3.2.3"
    "mujoco-python-viewer==0.1.4"
    "gymnasium==0.29.1"
    "stable_baselines3==2.4.0a8"
    "mujoco-mjx==3.3.0"
)

# Install each package
for package in "${packages[@]}"; do
    echo "Installing $package..."
    python3.12 pip install "$package"
    if [ $? -ne 0 ]; then
        echo "Failed to install $package"
        exit 1
    fi
done

echo "All packages installed successfully."
