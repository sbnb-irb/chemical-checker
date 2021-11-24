FROM conda/miniconda3-centos7:latest
# Steps to create/update the docker image in the gitlab server (coelho)
# cd /root/chemical_checker
# git pull --rebase
# cd /root/chemical_checker/container/docker
# sudo docker build -t cc_image .
RUN  yum install -y gcc g++ gcc-c++ cmake gfortran gcc-gfortran make wget squashfs-tools
RUN  yum install -y libarchive-devel
RUN  yum groupinstall 'Development Tools' -y
RUN  export SINGVER=3.6.4 && \
     wget https://github.com/hpcng/singularity/releases/download/v${SINGVER}/singularity-${SINGVER}.tar.gz && \
     tar -xvf singularity-${SINGVER}.tar.gz && \
     rm singularity-${SINGVER}.tar.gz
RUN  export GOVER=1.13 && \
     wget https://go.dev/dl/go$GOVER.linux-amd64.tar.gz && \
     tar -C /usr/local -xzvf go$GOVER.linux-amd64.tar.gz && \
     rm go$GOVER.linux-amd64.tar.gz
RUN  echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
     echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
     source ~/.bashrc &&  \
     cd singularity && \
     ./mconfig && \
     make -C ./builddir && \
     make -C ./builddir install
WORKDIR singularity
RUN  pip install -U setuptools twine
