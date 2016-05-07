#!/bin/bash
# Install's cuda and it's dependencies on an ubuntu 14.04 system.
# Largely taken from steps outlined at:
#  https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN)

echo -e "\nCUDA installation script\n"

# Script must be run as root
if [ "$(id -u)" != "0" ]; then
    echo "Please re-run this script as the root user." 1>&2
    exit 1
fi

# Ensure the system requirements are met
system=$(lsb_release -i | awk '{print $3}')
version=$(lsb_release -r | awk '{print $2}')

if [[ $system != 'Ubuntu' || $version != "14.04" ]]; then
    echo "This script needs to be run on Ubuntu 14.04" 1>&2
    exit 1
fi

echo -e "Installing build-essential if not already done\n"
apt-get -y install build-essential

echo -e "Downloading CUDA Web Installer\n"
curl -o /tmp/cuda-repo.deb "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb"

echo -e "Adding downloaded repository to repo list\n"
dpkg -i /tmp/cuda-repo.deb

echo -e "Installing CUDA - this should take about 10 minutes\n"
apt-get update
apt-get -y install cuda

echo -e "Updating the image for NVIDIA driver compatibility\n"
apt-get -y install linux-image-extra-virtual

echo -e "Blacklisting Nouveau\n"
cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOF

echo options nouveau modeset=0 > /etc/modprobe.d/nouveau-kms.conf
update-initramfs -u

echo -e "\nInstallation is complete.\nPlease restart the system before continuing.\n"
