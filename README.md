# :warning: Note for Editors and Reviewers

This repository is the one currently being used to develop the Chemical Checker (CC) in our SB&NB laboratory. As such, the repository contains a significant number of functionalities and data not presented in the primary CC manuscript.

Due to the strong computational requirements of our pipeline, the code has been written and optimized to work in our local HPC facilities. Installation guides found below are for SB&NB users only. As stated in the manuscript, the main deliverable of our resource are the CC _signatures_, which can be accessed easily through a [REST API](https://chemicalchecker.com/help) or downloaded as [data files](https://chemicalchecker.com/downloads).

Please feel free to explore any of the scripts inside this repository. The most relevant to the paper being revised are found under:
* Data pre-processing: `packages/scripts/preprocess/`.
* Signature production: `package/chemicalchecker/core/{sign1.py sign2.py}`.

# Chemical Checker

The Chemical Checker (CC) is a resource of small molecule signatures. In the CC, compounds are described from multiple viewpoints, spanning every aspect of the drug discovery process, from chemical properties to clinical outcomes.

* For more information about this repositiory, please refer to our [Wiki page](http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/wikis/home).
* For a quick exploration of the resource, please visit the [CC web app](http://chemicalchecker.org).
* For full documentation of the python package, please see the [API doc](http://project-specific-repositories.sbnb-pages.irbbarcelona.pcb.ub.es/chemical_checker).
* Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://biorxiv.org/content/10.1101/745703v1).

## Quick start

To fetch signatures (without fancy CC package capabilities) the package can be installed directly via `pip` from our local PyPI server:

```bash
sudo pip install --index http://coelho.irbbarcelona.pcb.ub.es:3141/root/dev/ --trusted-host coelho.irbbarcelona.pcb.ub.es chemicalchecker
```

_N.B. Only bare minimum dependencies are installed along with the package_

## Working from a laptop

Firtst, check that you are connected to the SB&NB local network:
```bash
ping coelho.irb.pcb.ub.es
```
Then, mount the remote filesystem
```bash
sudo mkdir /aloy
chown <laptop_username>:<laptop_username> /aloy
sshfs <sbnb_username>@pac-one-head.irb.pcb.ub.es:/aloy /aloy
```
You can unmount the filesystem with:
```bash
fusermount -u /aloy
```

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

> In case of errors during this step, check Singularity [prerequisites](https://www.sylabs.io/guides/2.6/user-guide/installation.html#before-you-begin)!

2. Add bind paths to singulairty config file:

        sudo echo "bind path = /aloy/web_checker" >> /etc/singularity/singularity.conf


    2.1. Make sure that `/aloy/web_checker` is available on your workstation (e.g. `ls /aloy/web_checker` should give a list of directories) if **not**:

        mkdir /aloy/web_checker
        sudo echo "fs-paloy.irbbarcelona.pcb.ub.es:/pa_webchecker /aloy/web_checker       nfs     defaults,_netdev 0 0" >> /etc/fstab
        sudo mount -a


3. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

        sudo apt-get install git

4. Download the [setup_chemicalchecker.sh](setup_chemicalchecker.sh) script to your home folder:

        wget http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/raw/master/setup_chemicalchecker.sh

5. Run the script (this script will require to type your password) with:

        sh setup_chemicalchecker.sh


The setup_chemicalchecker script has created an alias in your ~/.bashrc so you can start the Chamical Checker with:

        source ~/.bashrc
        chemcheck


After the first run of this script you can **update** the Chemical Checker package with the following command:

        sh setup_chemicalchecker.sh -u
        
If you only want to change the *default* config file, run the script with the -e argument:

        sh setup_chemicalchecker.sh -e
    
## Usage


1. Run a Jupyter Notebook with:

        chemcheck

    2.1. Open your browser, paste the URL that the script has produced.

    2.2. Start a new notebook (on the top right jupyter page click New -> Python )

    2.3. Type `import chemicalchecker`

2. Run a shell within the image:

        chemcheck -s [-d <PATH_TO_SOURCE_CODE_ROOT>] [-c <PATH_TO_CONFIG_FILE>]
        
    3.1 Type `ipython`
    
    3.2 Type `import chemicalchecker`


## Adding a package or software to the image

1. You will have to enter the singularity sandbox

        cd ~/chemical_checker
        sudo singularity shell --writable sandbox

2. Install the package/software and exit the image

        pip install <package_of_your_dreams>
        exit

3. Re-generate the image:

        rm cc.simg
        sudo singularity build cc.simg sandbox

4. In case you make use of the HPC utility, remember to copy your newly generated image to a directory accessible by the queueing system and edit the config file (section PATH.SINGULARITY_IMAGE) accordingly e.g.:

        cp cc.simg /aloy/scratch/<yout_user>/cc.simg


## Adding a permanent dependency to the package

Not re-inventing the wheel is a great philosophy, but each dependency we introduce comes at the cost of maintainability. Double check that the module you want to add is the best option for doing what you want to do. Check that it is actively developed and that it supports Python 2 and 3. Test it thoroughly using the sandbox approach presented above. When your approach is mature you can happily add the new dependency to the package.

To do so you can add a `pip install <package_of_your_dreams>` line to the following files in container/singularity:

* cc-full.def (the definition file used by setup_chemicalchecker.sh)
* cc_py27.def (unit-testing Python 2 environment)
* cc_py36.def (unit-testing Python 3 environment)

Don't forget to also add a short comment on why and where this new dependency is used, also in the commit message. E.g. "Added dependency used in preprocessing for space B5.003". The idea is that whenever B5.003 is obsoleted we can also safely remove the dependency.