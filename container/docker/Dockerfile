FROM conda/miniconda3-centos7:latest
# Steps to create/update the docker image in the gitlab server (coelho)
# cd /root/chemical_checker
# git pull --rebase
# cd /root/chemical_checker/container/docker
# sudo docker build -t cc_image .
RUN  yum install -y gcc g++ gcc-c++ cmake gfortran gcc-gfortran make wget squashfs-tools
RUN  yum install -y libarchive-devel
RUN  yum groupinstall 'Development Tools' -y
RUN  export SINGVER=3.6.4
RUN  wget https://github.com/hpcng/singularity/releases/download/v3.6.4/singularity-3.6.4.tar.gz
RUN  tar -xvf singularity-3.6.4.tar.gz
WORKDIR "./singularity-3.6.4"
RUN  ./configure --prefix=/usr/local
RUN  make
RUN  make install
RUN  pip install -U setuptools twine
