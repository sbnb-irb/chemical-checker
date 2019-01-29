# Singularity CC Image

This def file allows to create a singularity image to use the Chemical Checker package

## Installation

1. Install singularity:  https://www.sylabs.io/guides/2.6/user-guide/installation.html
2. Create sandbox directory in your local machine:

        mkdir /path/to/sandbox
3. Create sandbox image from the def file with sudo:

        sudo singularity build --sandbox /path/to/sandbox /path/to/cc.def
4. Create image from sandbox:

        sudo singularity build cc.simg  /path/to/sandbox

## Usage

You can use the image by doing:


    singularity shell /path/to/image
    python
or

    singularity exec /path/to/image python


## Notebooks

Run this command in your desktop machine:

    singularity exec -B /home/<username>/run_user_sing:/run/user <path-to-image> jupyter notebook