# Chemical Checker

The Chemical Checker (CC) is a resource of small molecule signatures. In the CC, the realms of chemoinformatics and bioinformatics are unified and compounds are described from multiple viewpoints, spanning every aspect of the drug discovery process, from chemical properties to clinical outcomes.

* For more information about this repositiory, please refer to our [Wiki page](http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/wikis/home).
* For a quick exploration of the resource, please visit the [CC web app](http://chemicalchecker.org).
* For full API documentation of the python package [API Doc](http://project-specific-repositories.sbnb-pages.irbbarcelona.pcb.ub.es/chemical_checker)
* Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://www.dropbox.com/s/x2rqszfdfpqdqdy/duranfrigola_etal_ms_current.pdf?dl=0).

## Installation 

1. [Install singularity](https://www.sylabs.io/guides/2.6/user-guide/installation.html)

2. [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

3. Download the `setup_chemicalchecker.sh` script to your home folder

4. Run the script with:

        sh setup_chemicalchecker.sh

    This script will require sudo access and it will request to type your password.
    
`setup_chemicalchecker.sh` allows you to create a singularity image with all necessary packages including the Chemical Checker package.

After the first run of this script, if you only want to **update** the Chemical Checker package, run the script like that:

        sh setup_chemicalchecker.sh -i
        
Every time, you run this script it will rrequire you to modify or just validate the Chemical Checker config file.
If you only want to change the config file, run the script like that:

        sh setup_chemicalchecker.sh -e
    
## Usage

1. Download the `run_chemicalchecker.sh` script to your home folder

2. Run the script with (if you want to use it as notebook):

        sh run_chemicalchecker.sh

    2.1. Open your browser, paste the URL that the script has produced.

    2.2. Start a new notebook (on the top right jupyter page click New -> Python )

    2.3. Type `import chemicalchecker`

3. Run the script with:

        sh run_chemicalchecker.sh -s
        
    3.1 Type `python`
    
    3.2 Type `import chemicalchecker`

## Examples

For use case examples, please see notebooks in this repository.
