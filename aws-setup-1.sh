#!/bin/bash
# Install's the rest of the 

# Largely taken from steps outlined at:
#  https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN)

echo -e "\n  [x] CUDA installation script\n"

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

echo -e "\n  [x] Installing caffe binary dependencies\n"
apt-get update
apt-get -y install build-essential libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev python-yaml python-numpy python-opencv rabbitmq-server
easy_install pillow
cp tracking-server/rabbitmq.config /etc/rabbitmq/

echo -e "\n  [x] Installing caffe python dependencies (~20 mins)\n"
cat tracking-server/py-faster-rcnn/caffe-fast-rcnn/python/requirements.txt | xargs -L 1 pip install
pip install easydict pika pafy
sed -i 's/, unquote_plus/\n    from urllib import unquote_plus/g' /usr/local/lib/python2.7/dist-packages/pafy/backend_internal.py
rm /usr/local/lib/python2.7/dist-packages/pafy/backend_internal.pyc

echo -e "\n  [x] Adding drivers and downloaded repository to repo list\n"
add-apt-repository -y ppa:graphics-drivers/ppa

echo -e "\n  [x] Blacklisting Nouveau\n"
cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOF

echo options nouveau modeset=0 > /etc/modprobe.d/nouveau-kms.conf
update-initramfs -u

echo -e "\n  [x] First step of installation is complete.\n  [x] Please restart the system before continuing with install-2.sh.\n"
