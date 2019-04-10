# Chemical Checker

The Chemical Checker (CC) is a resource of small molecule signatures. In the CC, the realms of chemoinformatics and bioinformatics are unified and compounds are described from multiple viewpoints, spanning every aspect of the drug discovery process, from chemical properties to clinical outcomes.

* For more information about this repositiory, please refer to our [Wiki page](http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/wikis/home).
* For a quick exploration of the resource, please visit the [CC web app](http://chemicalchecker.org).
* For full API documentation of the python package [API Doc](http://project-specific-repositories.sbnb-pages.irbbarcelona.pcb.ub.es/chemical_checker)
* Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://www.dropbox.com/s/x2rqszfdfpqdqdy/duranfrigola_etal_ms_current.pdf?dl=0).

## Working from a laptop

1. Check that you are connected to the local network

        ping coelho.irb.pcb.ub.es

2. Mount the remote filesystem

        sudo mkdir /aloy
        chown <laptop_username>:<laptop_username> /aloy
        sshfs <sbnb_username>@pac-one-head.irb.pcb.ub.es:/aloy /aloy

You can unmount the filesystem with:

        fusermount -u /aloy


## Quick start

To fetch signatures (without fancy CC package capabilities) the package can be installed directly via `pip` from our local PyPI server:

```shell
sudo pip install --index http://coelho.irbbarcelona.pcb.ub.es:3141/root/dev/ --trusted-host coelho.irbbarcelona.pcb.ub.es chemicalchecker
```

_N.B. Only bare minimum dependencies are installed along with the package_

## Complete Installation 

For an advanced usage of the CC package capabilities, we recomend creating the CC dependency enviroment within a container image:

1. [Install Singularity](https://www.sylabs.io/guides/2.6/user-guide/installation.html)

        VER=2.5.1
        wget https://github.com/sylabs/singularity/releases/download/$VER/singularity-$VER.tar.gz
        tar xvf singularity-$VER.tar.gz
        cd singularity-$VER
        ./configure --prefix=/usr/local --sysconfdir=/etc
        make
        sudo make install

2. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

        apt-get install git

3. Download the [setup_chemicalchecker.sh](setup_chemicalchecker.sh) script to your home folder:

        wget http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/raw/master/setup_chemicalchecker.sh

4. Run the script (this script will require sudo access and it will request to type your password) with:

        sh setup_chemicalchecker.sh


The setup_chemicalchecker script has created an alias in your ~/.bashrc so you can start the Chamical Checker with:

        source ~/.bashrc
        chemcheck


After the first run of this script you can **update** the Chemical Checker package with the following command:

        sh setup_chemicalchecker.sh -i
        
If you only want to change the config file, run the script with the -e argument:

        sh setup_chemicalchecker.sh -e
    
## Usage


1. Run a Jupiter Notebook with:

        chemcheck

    2.1. Open your browser, paste the URL that the script has produced.

    2.2. Start a new notebook (on the top right jupyter page click New -> Python )

    2.3. Type `import chemicalchecker`

2. Run a shell within the image:

        chemcheck -s
        
    3.1 Type `ipython`
    
    3.2 Type `import chemicalchecker`


## Adding a package or software to the image

1. You will have to enter the singularity sandbox

        cd ~/local_checker
        sudo singularity shell --writable sandbox

2. Install the package/software and exit the image

        pip install <package_of_your_dreams>
        exit

3. Re-generate the image:

        rm cc.simg
        sudo singularity build cc.simg sandbox

## Examples

For use case examples, please see notebooks in this repository.
