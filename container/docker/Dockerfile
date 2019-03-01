FROM conda/miniconda3-centos7:latest
# Steps to create the docker image in the gitlab server(coelho)
# Copy this file to a root access directory
# cd to that directory
# sudo docker build -t cc_image .
RUN  yum install -y gcc g++ gcc-c++ cmake gfortran gcc-gfortran make wget
RUN  yum install -y libarchive-devel
RUN  yum groupinstall 'Development Tools' -y
RUN  export SINGVER=2.5.1
RUN  wget https://github.com/sylabs/singularity/releases/download/2.5.1/singularity-2.5.1.tar.gz
RUN  tar xvf singularity-2.5.1.tar.gz
WORKDIR "./singularity-2.5.1"
RUN  ./configure --prefix=/usr/local
RUN  make
RUN  make install