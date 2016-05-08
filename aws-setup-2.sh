#!/bin/bash
# Install's the rest of the dependencies for Caffe and the tracking application.

# Largely taken from steps outlined at:
#  https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN)

echo -e "\n  [x] Snooker Tracking Installation Script Pt. 2\n"

# Script must be run as root
if [ "$(id -u)" != "0" ]; then
    echo "Please re-run this script as the root user." 1>&2
    exit 1
fi

# Ensure the system requirements are met
system=$(lsb_release -i | awk '{print $3}')
version=$(lsb_release -r | awk '{print $2}')
user=$(who am i | awk '{print $1}')

if [[ $system != 'Ubuntu' || $version != "14.04" ]]; then
    echo "This script needs to be run on Ubuntu 14.04" 1>&2
    exit 1
fi

echo -e "\n  [x] Installing NVIDIA drivers\n"
apt-get update
apt-get -y install linux-source linux-headers-`uname -r`
apt-get -y install nvidia-364 nvidia-364-dev
modprobe nvidia

echo -e "\n  [x] Downloading CUDA Web Installer\n"
curl -o /tmp/cuda-repo.deb "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb"
dpkg -i /tmp/cuda-repo.deb
apt-get update

echo -e "\n  [x] Installing CUDA (~10 mins)\n"
apt-get -y install cuda

echo -e "\n  [x] Updating PATH variables in .bashrc\n"
echo 'export PATH=$PATH:/usr/local/cuda/bin' >> /home/$user/.bashrc
echo 'export LD_LIBRARY_PATH=:/usr/local/cuda/lib64' >> /home/$user/.bashrc
source /home/$user/.bashrc

echo -e "\n  [x] Installing Caffe dependencies (~20 mins)\n"
apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev python-yaml python-numpy python-opencv rabbitmq-server
easy_install pillow

cd tracking-server/py-faster-rcnn/caffe-fast-rcnn/
cat python/requirements.txt | xargs -L 1 pip install

echo -e "\n  [x] Bulding configuration file for Caffe\n"
cat > Makefile.config << EOF
CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_50,code=compute_50
BLAS := atlas
PYTHON_INCLUDE := /usr/include/python2.7 \
                    /usr/lib/python2.7/dist-packages/numpy/core/include
PYTHON_LIB := /usr/lib
WITH_PYTHON_LAYER := 1
INCLUDE_DIRS := \$(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := \$(PYTHON_LIB) /usr/local/lib /usr/lib
BUILD_DIR := build
DISTRIBUTE_DIR := distribute
TEST_GPUID := 0
Q ?= @
EOF

echo -e "\n  [x] Installing Caffe (~10 mins)\n"
make -j8 && make pycaffe

cd ../lib
# Fix the architecture issue with AWS
sed -i 's/arch=sm_../arch=sm_30/g' setup.py
make

pip install easydict pika pafy

# Issue with pafy not using the correct libraries
sed -i 's/, unquote_plus/\n    from urllib import unquote_plus/g' /usr/local/lib/python2.7/dist-packages/pafy/backend_internal.py
rm /usr/local/lib/python2.7/dist-packages/pafy/backend_internal.pyc

echo -e "\n  [x] Configuring the rabbitmq server\n"
cd ../../
chown -R $user:$user ./*
cp rabbitmq.config /etc/rabbitmq/
rabbitmqctl reset


echo -e "\n  [x] Installation is complete. Please restart and run the tracking server."

