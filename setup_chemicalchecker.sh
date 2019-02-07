#!/bin/sh

CC_HOME=$HOME/chemical_checker

REMOTE_CCREPO=http://gitlab.sbnb.org/project-specific-repositories/chemical_checker.git
LOCAL_CCREPO=$CC_HOME/chemical_checker

REMOTE_IMAGE=/aloy/home/mbertoni/images/cc.simg
LOCAL_IMAGE=$LOCAL_CCREPO/container/singularity/cc.simg

JUPYTER_DIR=$CC_HOME/run_user_sing

# check if needed binaries are available
_=$(command -v singularity);
if [ "$?" != "0" ]; then
    printf -- "\033[31m ERROR: You don't seem to have Singularity installed \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 127;
fi;

_=$(command -v git);
if [ "$?" != "0" ]; then
    printf -- "\033[31m ERROR: You don't seem to have Git installed \033[0m\n";
    printf -- 'Follow the guide at: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git\n';
    exit 127;
fi;

# check if the chemical checker repository is available
if [ -d "$LOCAL_CCREPO" ]
then
    printf -- '\033[32m SUCCESS: Chemical Checker repository available. (%s) \033[0m\n' $LOCAL_CCREPO;
else
    printf -- 'cloning Chemical Checker package from GitLab... \n';
    git clone $REMOTE_CCREPO $LOCAL_CCREPO;
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: repository cloned. \033[0m\n';
    else
        printf -- '\033[31m ERROR: could not clone repository. \033[0m\n';
        exit 1;
    fi
fi

# check if a local singularity image is available, otherwise copy
if [ -f "$LOCAL_IMAGE" ]
then
    printf -- '\033[32m SUCCESS: Singularity image available. (%s) \033[0m\n' $LOCAL_IMAGE;
else
    printf -- 'synching singularity image... \n';
    rsync --info=progress2 $REMOTE_IMAGE $LOCAL_IMAGE;
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: image copied. \033[0m\n';
    else
        printf -- '\033[31m ERROR: could copy image. \033[0m\n';
        exit 2;
    fi
fi

# preapare jupyter notebook dir
if [ -d "$JUPYTER_DIR" ]
then
    printf -- '\033[32m SUCCESS: Jupyter Notebook directory available. (%s) \033[0m\n' $JUPYTER_DIR;
else
    mkdir $CC_HOME/run_user_sing
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: Jupyter Notebook directory created. \033[0m\n';
    else
        printf -- '\033[31m ERROR: could not create Jupyter Notebook directory. \033[0m\n';
        exit 3;
    fi
    chmod +w $CC_HOME/run_user_sing
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: Jupyter Notebook directory permissions. \033[0m\n';
    else
        printf -- '\033[31m ERROR: could not set permission for Jupyter Notebook directory. \033[0m\n';
        exit 4;
    fi
fi

printf -- 'Starting Jupyter Notebook... \n';
SINGULARITYENV_PYTHONPATH=$LOCAL_CCREPO/chemicalchecker \
SINGULARITYENV_CC_CONFIG=$LOCAL_CCREPO/cc_config.json \
singularity exec --cleanenv -B $CC_HOME/run_user_sing:/run/user $LOCAL_IMAGE jupyter notebook

