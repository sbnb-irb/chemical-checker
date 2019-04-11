#!/bin/sh

LOCAL_CCREPO=$HOME/chemical_checker

LOCAL_IMAGE=$LOCAL_CCREPO/cc.simg

JUPYTER_DIR=$LOCAL_CCREPO/run_user_sing

# display usage for current script
usage () {
    echo ""
    echo "Run an interactive Python session as Jupiter Notebook."
    echo "Usage: $ bash run_chemicalchecker.sh [-sh]"
    echo ""
    echo "  -s      launch simple shell in the image"
    echo "  -d      use external Chemical Checker (develop mode)"
    echo "  -c      use external Chemical Checker config file"
    echo "  -h      print this help"
    echo ""
    exit 1
}

# parse arguments
unset SINGULARITY_SHELL
SINGULARITY_SHELL=false
EXTERNAL_CCREPO=false
EXTERNAL_CCCONFIG=false
while getopts ':sc:d:hD' c
do
  case $c in
    s) SINGULARITY_SHELL=true ;;
    d) EXTERNAL_CCREPO=true; PATH_CCREPO=$OPTARG ;;
    c) EXTERNAL_CCCONFIG=true; PATH_CCCONFIG=$OPTARG ;;
    D) DEBUG=true ;;
    h) usage ;; esac
done

# print variables if debugging
if [ "$DEBUG" = true ]
then
    echo NAME $NAME
    echo OPTARG $OPTARG
    echo SINGULARITY_SHELL $SINGULARITY_SHELL;
    echo EXTERNAL_CCREPO $EXTERNAL_CCREPO;
    echo PATH_CCREPO $PATH_CCREPO;
    echo EXTERNAL_CCCONFIG $EXTERNAL_CCCONFIG;
    echo PATH_CCCONFIG $PATH_CCCONFIG;
    exit 1;
fi

# check if valid path
if [ "$OPTARG" = "d" ] && [ ! -d "$PATH_CCREPO" ]
then
    printf -- "\033[31m ERROR: You need to specify a valid path when using the -d option. \033[0m\n";
    exit 1;
fi
if [ "$OPTARG" = "c" ] && [ ! -f "$PATH_CCCONFIG" ]
then
    printf -- "\033[31m ERROR: You need to specify a valid file when using the -d option. \033[0m\n";
    exit 1;
fi

# check if singularity is available
_=$(command -v singularity);
if [ "$?" != "0" ]; then
    printf -- "\033[31m ERROR: You don't seem to have Singularity installed \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 127;
fi;

# check if a local singularity image is available
if [ -f "$LOCAL_IMAGE" ]
then
    printf -- '\033[32m SUCCESS: Singularity image available. (%s) \033[0m\n' $LOCAL_IMAGE;
else
    printf -- '\033[31m ERROR: image not found. \033[0m\n';
fi

# using external package or config or both?
if [ "$EXTERNAL_CCREPO" = false ]
then
    PATH_CCREPO=/opt/chemical_checker/package/
fi
if [ "$EXTERNAL_CCCONFIG" = false ]
then
    PATH_CCCONFIG=/opt/chemical_checker/cc_config.json
fi

# run shell or notebook?
if [ "$SINGULARITY_SHELL" = true ]
then
    printf -- 'Starting Singularity Shell... (Press CTRL+D to exit)\n';
    SINGULARITYENV_PYTHONPATH=$PATH_CCREPO SINGULARITYENV_CC_CONFIG=$PATH_CCCONFIG singularity shell --cleanenv $LOCAL_IMAGE;
else
    printf -- 'Starting Jupyter Notebook... (Press CTRL+C to terminate)\n';
    # preapare jupyter notebook dir
    if [ ! -d "$JUPYTER_DIR" ]
    then
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
    SINGULARITYENV_PYTHONPATH=$PATH_CCREPO SINGULARITYENV_CC_CONFIG=$PATH_CCCONFIG singularity exec --cleanenv -B $JUPYTER_DIR:/run/user $LOCAL_IMAGE jupyter notebook;
fi
