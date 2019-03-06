#!/bin/sh

LOCAL_CCREPO=$HOME/local_checker

LOCAL_IMAGE=$LOCAL_CCREPO/cc.simg


JUPYTER_DIR=$LOCAL_CCREPO/run_user_sing

# display usage for current script
usage () {
    echo ""
    echo "Usage: $ bash run_chemicalchecker.sh [-sh]"
    echo ""
    echo "  -s      launch singularity shell (instead of jupyter notebook)"
    echo "  -h      print this help"
    echo ""
    exit 1
}

# parse arguments
unset SINGULARITY_SHELL
SINGULARITY_SHELL=false
while getopts 'sh' c
do
  case $c in
    s) SINGULARITY_SHELL=true ;;
    h) usage ;; esac
done

# check if needed binaries are available
_=$(command -v singularity);
if [ "$?" != "0" ]; then
    printf -- "\033[31m ERROR: You don't seem to have Singularity installed \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 127;
fi;


# check if a local singularity image is available, otherwise copy
if [ -f "$LOCAL_IMAGE" ]
then
    printf -- '\033[32m SUCCESS: Singularity image available. (%s) \033[0m\n' $LOCAL_IMAGE;
else
    printf -- '\033[31m ERROR: image not found. \033[0m\n';
fi

# preapare jupyter notebook dir
if [ -d "$JUPYTER_DIR" ]
then
    printf -- '\033[32m SUCCESS: Jupyter Notebook directory available. (%s) \033[0m\n' $JUPYTER_DIR;
else
    mkdir $JUPYTER_DIR
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: Jupyter Notebook directory created. \033[0m\n';
    else
        printf -- '\033[31m ERROR: could not create Jupyter Notebook directory. \033[0m\n';
        exit 3;
    fi
    chmod +w $JUPYTER_DIR
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: Jupyter Notebook directory permissions. \033[0m\n';
    else
        printf -- '\033[31m ERROR: could not set permission for Jupyter Notebook directory. \033[0m\n';
        exit 4;
    fi
fi

if [ "$SINGULARITY_SHELL" = true ]
then
    printf -- 'Starting Singularity Shell... (Press CTRL+D to exit)\n';
    singularity shell --cleanenv $LOCAL_IMAGE
else
    printf -- 'Starting Jupyter Notebook... \n';
    singularity exec --cleanenv -B $JUPYTER_DIR:/run/user $LOCAL_IMAGE jupyter notebook
fi
