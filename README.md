# Chemical Checker

The Chemical Checker (CC) is a resource of small molecule signatures. In the CC, the realms of chemoinformatics and bioinformatics are unified and compounds are described from multiple viewpoints, spanning every aspect of the drug discovery process, from chemical properties to clinical outcomes.

* For more information about this repositiory, please refer to our [Wiki page](http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/wikis/home).
* For a quick exploration of the resource, please visit the [CC web app](http://chemicalchecker.org).
* Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://www.dropbox.com/s/x2rqszfdfpqdqdy/duranfrigola_etal_ms_current.pdf?dl=0).

## How to start 

1. Install singularity:  https://www.sylabs.io/guides/2.6/user-guide/installation.html

2. Download def file

    Go to this link and use the download cloud button to get the def file
    http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/blob/master/container/singularity/cc-full.def
    
3. Create singularity sandbox

        sudo singularity build --sanbox <PATH_TO_SANDBOX_DIRECTORY> <PATH_TO_>/cc-full.def

    
4. Modify config file with your specific data

    Run this command and change the paramaters:
    
        sudo singularity --writable <PATH_TO_SANDBOX_DIRECTORY> vi /opt/chemical_checker/cc_config.json
    
5. Create final container image

        sudo singularity build cc.simg <PATH_TO_SANDBOX_DIRECTORY>

## Installation

1. Download the `setup_chemicalchecker.sh` script to your home folder

2. run the script with:
    ```sh setup_chemicalchecker.sh```

4. Open your browser to [http://localhost:8888/](http://localhost:8888/).

5. Start a new notebook (on the top right jupyter page click New -> Python )

6. ```import chemicalchecker```

7. Have fun!
