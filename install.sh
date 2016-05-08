#!/bin/bash
# Install's the rest of the 

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
user=$(who am i | awk '{print $1}')

if [[ $system != 'Ubuntu' || $version != "14.04" ]]; then
    echo "This script needs to be run on Ubuntu 14.04" 1>&2
    exit 1
fi

echo -e "Installing build-essential if not already done\n"
apt-get update
apt-get -y install build-essential

echo -e "Adding drivers and downloaded repository to repo list\n"
add-apt-repository -y ppa:graphics-drivers/ppa

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

echo -e "\nFirst step of installation is complete.\nPlease restart the system before continuing.\n"
