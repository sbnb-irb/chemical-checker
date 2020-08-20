# Chemical Checker Repository

The **Chemical Checker (CC)** is a resource of small molecule signatures. In the CC, compounds are described from multiple viewpoints, spanning every aspect of the drug discovery pipeline, from chemical properties to clinical outcomes.

* For a quick exploration of what this resource enables, please visit the [CC web app](http://chemicalchecker.org).
* For full documentation of the python package, please see the [Documentation](http://packages.sbnb-pages.irbbarcelona.org/chemical_checker).
* Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://biorxiv.org/content/10.1101/745703v1).
* For more information about this repository, discussion, notes, etc... please refer to our [Wiki page](http://gitlabsbnb.irbbarcelona.org/packages/chemical_checker/wikis/home).

The **Chemical Checker Repository** holds the current implementation of the CC in our `SB&NB` laboratory. As such, the repository contains a significant number of functionalities and data not presented in the primary CC manuscript. The repository follows its directory structure:

* `container`: Deal with containerization of the CC. It contains the definition files for Singularity image.
* `notebook`: Contains exemplary Jupyter Notebooks that showcase some CC features.
* `package`: The backbone of the CC in form of a Python package.
* `pipelines`: The pipeline script for update and generation of data for the web app.


Due to the strong computational requirements of our pipeline, the code has been written and optimized to work in our local HPC facilities. Installation guides found below are mainly addressed to `SB&NB` users. As stated in the manuscript, the main deliverable of our resource are the CC _signatures_, which can be easily accessed:

* through a [REST API](https://chemicalchecker.com/help),
* downloaded as [data files](https://chemicalchecker.com/downloads) or 
* predicted from SMILES with the [Signaturizer](http://gitlabsbnb.irbbarcelona.org/packages/signaturizer).

## Chemical Checker `lite`

The CC package can be installed directly via `pip` from our local PyPI server:

```bash
sudo pip install --index http://gitlabsbnb.irbbarcelona.org:3141/root/dev/ --trusted-host gitlabsbnb.irbbarcelona.org chemicalchecker
```

This installs the `lite` version of the CC that can be used for basic task (e.g. to open signatures) but most of the fancy CC package capabilities will be missing.

_**N.B.** Only bare minimum dependencies are installed along with the package_



## Dependencies

All the dependencies for the CC will be bundled within a singularity image generated during the installation process.
However, to generate such an image we require some software being available:

### Singularity

1. [Install Singularity](https://www.sylabs.io/guides/2.6/user-guide/installation.html)

        VER=2.5.1
        wget https://github.com/sylabs/singularity/releases/download/$VER/singularity-$VER.tar.gz
        tar xvf singularity-$VER.tar.gz
        cd singularity-$VER
        ./configure --prefix=/usr/local --sysconfdir=/etc
        make
        sudo make install

> In case of errors during this step, check Singularity [prerequisites](https://www.sylabs.io/guides/2.6/user-guide/installation.html#before-you-begin)!



2. Add bind paths to singularity config file:

        sudo echo "bind path = /aloy/web_checker" >> /etc/singularity/singularity.conf


3. Make sure that `/aloy/web_checker` is available on your workstation (e.g. `ls /aloy/web_checker` should give a list of directories) if **not**:

        mkdir /aloy/web_checker
        sudo echo "fs-paloy.irbbarcelona.pcb.ub.es:/pa_webchecker /aloy/web_checker       nfs     defaults,_netdev 0 0" >> /etc/fstab
        sudo mount -a

### Git

1. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

        sudo apt-get install git


## Installation 

For an advanced usage of the CC package capabilities, we recommend creating the CC dependency environment within a container image:


1. Clone this repository to your code folder:
        
        cd ~ && mkdir -p code && cd code
        git clone http://gitlabsbnb.irbbarcelona.org/packages/chemical_checker.git

2. Run the script (this script will require to type your password) with:

        cd chemical_checker && sh setup/setup_chemicalchecker.sh

### Running `Vanilla` Chemical Checker

This is the easiest scenario where you simply use the CC code 'as is'.

The setup_chemicalchecker script has created an alias in your ~/.bashrc so you can start the CC with:
```bash
source ~/.bashrc
chemcheck
```

### Running custom Chemical Checker

If you are contributing with code to the CC you can run the singularity image specifying your local develop package branch:
```bash
chemcheck -d /path/to/your/code/chemical_checker/package/
```
    
### Running with alternative config file

The CC rely on one config file containing the information for the current enviroment (e.g. the HPC, location of the default CC, database, etc...). The default configuration can be overridden specifying a custom config file:
```bash
chemcheck -c /path/to/your/cc_config.json
```

## Usage

We make it trivial to either start a Jupyter Notebook within the image or to run a shell:

1. Run a Jupyter Notebook with:

        chemcheck

    1.1. Open your browser, paste the URL that the script has produced.

    1.2. Start a new notebook (on the top right Jupyter page click New -> Python )

    1.3. Type `import chemicalchecker`

2. Run a shell within the image:

        chemcheck -s [-d <PATH_TO_SOURCE_CODE_ROOT>] [-c <PATH_TO_CONFIG_FILE>]
        
    2.1 Type `ipython`
    
    2.2 Type `import chemicalchecker`


## Introducing new dependencies

### Adding a package or software to the image

1. You will have to enter the singularity sandbox

        cd ~/chemical_checker
        sudo singularity shell --writable sandbox

2. Install the package/software and exit the image

        pip install <package_of_your_dreams>
        exit

3. Re-generate the image:

        rm cc.simg
        sudo singularity build cc.simg sandbox

4. In case you make use of the HPC utility, remember to copy your newly generated image to a directory accessible by the queuing system and edit the config file (section PATH.SINGULARITY_IMAGE) accordingly e.g.:

        cp cc.simg /aloy/scratch/<yout_user>/cc.simg


### Adding a permanent dependency to the package

Not re-inventing the wheel is a great philosophy, but each dependency we introduce comes at the cost of maintainability. Double check that the module you want to add is the best option for doing what you want to do. Check that it is actively developed and that it supports Python 3. Test it thoroughly using the sandbox approach presented above. When your approach is mature you can happily add the new dependency to the package.

To do so you can add a `pip install <package_of_your_dreams>` line to the following files in container/singularity:

* cc-full.def (the definition file used by setup_chemicalchecker.sh)
* cc_py36.def (unit-testing Python 3 environment)

Don't forget to also add a short comment on why and where this new dependency is used, also in the commit message. E.g. "Added dependency used in preprocessing for space B5.003". The idea is that whenever B5.003 is obsoleted we can also safely remove the dependency.

## `SB&NB` configuration

### Working from a laptop

First, check that you are connected to the `SB&NB` local network:
```bash
ping pac-one-head.irb.pcb.ub.es
```
Then, mount the remote filesystem
```bash
sudo mkdir /aloy
chown <laptop_username>:<laptop_username> /aloy
sshfs <sbnb_username>@pac-one-head.irb.pcb.ub.es:/aloy /aloy
```
You can unmount the filesystem with:
```bash
# Linux
fusermount -u /aloy
# MacOSX
umount /aloy
```
