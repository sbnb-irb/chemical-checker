#!/bin/sh

LOCAL_CCREPO=$HOME/chemical_checker

REMOTE_CCREPO=http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/raw/master/

LOCAL_IMAGE=$LOCAL_CCREPO/cc.simg

LOCAL_IMAGE_SANDBOX=$LOCAL_CCREPO/sandbox

# display usage for current script
usage () {
    echo ""
    echo "Install the Chemical Checker in $LOCAL_CCREPO."
    echo "Usage: $ bash $0 [-ueh] [-d path_to_CC_package]"
    echo ""
    echo "  -u      update chemical checker"
    echo "  -e      edit config file"
    echo "  -d      path to source code (development mode)"
    echo "  -h      print this help"
    echo ""
    exit 1
}

# compare versions
version_gt () {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1";
}

# parse arguments
unset CREATE_IMAGE UPDATE_IMAGE EDIT_CONFIG CHANGE_ENV UPDATE_CC
CREATE_IMAGE=true
UPDATE_IMAGE=false
EDIT_CONFIG=false
CHANGE_ENV=false
UPDATE_CC=false
while getopts ':ued:hD' c
do
  case $c in
    u) CREATE_IMAGE=false; UPDATE_IMAGE=true; UPDATE_CC=true ;;
    e) CREATE_IMAGE=false; UPDATE_IMAGE=true; EDIT_CONFIG=true ;;
    d) CREATE_IMAGE=false; UPDATE_IMAGE=true; CHANGE_ENV=true; PATH_BRANCH=$OPTARG ;;
    D) DEBUG=true ;;
    h) usage ;; esac
done

# print variables if debugging
if [ "$DEBUG" = true ]
then
    echo NAME $NAME
    echo OPTARG $OPTARG
    echo CREATE_IMAGE $CREATE_IMAGE;
    echo UPDATE_IMAGE $UPDATE_IMAGE;
    echo EDIT_CONFIG $EDIT_CONFIG;
    echo CHANGE_ENV $CHANGE_ENV;
    echo UPDATE_CC $UPDATE_CC;
    echo PATH_BRANCH $PATH_BRANCH;
    exit 1;
fi

# check if valid path
if [ "$OPTARG" = "d" ] && [ ! -d "$PATH_BRANCH" ]
then
    printf -- "\033[31m ERROR: You need to specify a valid path when using the -d option. \033[0m\n";
    exit 1;
fi

# check if singularity is available
_=$(command -v singularity);
if [ "$?" != "0" ]
then
    printf -- "\033[31m ERROR: You don't seem to have Singularity installed \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 1;
fi;

# check singularity version
SINGULARITY_MIN_VERSION=2.5.0
SINGULARITY_INSTALLED_VERSION="$(singularity --version)"
if version_gt $SINGULARITY_MIN_VERSION $SINGULARITY_INSTALLED_VERSION
then
    printf -- "\033[31m ERROR: Update Singularity, we require at least version ${SINGULARITY_MIN_VERSION} (${SINGULARITY_INSTALLED_VERSION} detected) \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 2;
fi


# check if git is available
_=$(command -v git);
if [ "$?" != "0" ]
then
    printf -- "\033[31m ERROR: You don't seem to have Git installed \033[0m\n";
    printf -- 'Follow the guide at: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git\n';
    exit 3;
fi;




# if we are creating the image we remove image and sandbox
if [ "$CREATE_IMAGE" = true ]
then
    if [ -d "$LOCAL_CCREPO" ]
    then
        printf -- '\033[33m WARNING: %s already exists, delete it to proceed. \033[0m\n' $LOCAL_CCREPO;
        exit 0;
    fi
    mkdir $LOCAL_CCREPO;
    cd $LOCAL_CCREPO;
    # download the definition file
    sudo rm -f cc-full.def;
    wget $REMOTE_CCREPO/container/singularity/cc-full.def;
    if [ $? -eq 0 ]
    then
        printf -- '\033[32m SUCCESS: Singularity definition file downloaded. \033[0m\n';
    else
        printf -- '\033[31m ERROR: Could not download definition file. \033[0m\n';
        exit 4;
    fi

    printf -- 'Removing old singularity image...\n';
    sudo rm -f $LOCAL_IMAGE;
    sudo rm -rf $LOCAL_IMAGE_SANDBOX;

    printf -- 'Creating singularity sandbox image... \n';
    sudo singularity build --sandbox $LOCAL_IMAGE_SANDBOX cc-full.def;
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: Image sandbox created correctly. \033[0m\n';
    else
        printf -- '\033[31m ERROR: Cannot create sandbox image. \033[0m\n';
        exit 5;
    fi

    # generate image from sandbox
    sudo singularity build $LOCAL_IMAGE $LOCAL_IMAGE_SANDBOX;
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: Image file created. \033[0m\n';
    else
        printf -- '\033[31m ERROR: Cannot create image. \033[0m\n';
        exit 6;
    fi

    # add alias to bashrc
    sudo rm -f run_chemicalchecker.sh;
    wget $REMOTE_CCREPO/run_chemicalchecker.sh;
    echo "alias chemcheck=\"sh ${LOCAL_CCREPO}/run_chemicalchecker.sh\"" >> ~/.bashrc;

fi

# check if a local singularity image is available, otherwise copy
if [ "$UPDATE_IMAGE" = true ]
then
    cd $LOCAL_CCREPO;
    # remove the previous image
    sudo rm -f $LOCAL_IMAGE;

    # change pythonpath in the image permanently to different CC repository
    # if [ "$CHANGE_ENV" = true ]
    # then
    #     text_replace_py="/export PYTHONPATH/c\\    export PYTHONPATH=\""$PATH_BRANCH"/package\":\$PYTHONPATH";
    #     text_replace_conf="/export CC_CONFIG/c\\    export CC_CONFIG=\""$PATH_BRANCH"/cc_config.json\"";
    #     sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX sed -i "$text_replace_py" /environment;
    #     sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX sed -i "$text_replace_conf" /environment;
    # fi

    # update CC to latest
    if [ "$UPDATE_CC" = true ]
    then
        # update sandbox
        sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX git --git-dir=/opt/chemical_checker/.git pull;
        if [ $? -eq 0 ]; then
            printf -- '\033[32m SUCCESS: Pulled latest Chemical Checker source code. \033[0m\n';
        else
            printf -- '\033[31m ERROR: Cannot update sandbox image. \033[0m\n';
            exit 7;
        fi
    fi

    # modify the config file
    if [ "$EDIT_CONFIG" = true ]
    then
        cd $LOCAL_CCREPO;
        sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX vi /opt/chemical_checker/cc_config.json
        # generate image from sandbox
        sudo singularity build $LOCAL_IMAGE $LOCAL_IMAGE_SANDBOX
        if [ $? -eq 0 ]; then
            printf -- '\033[32m SUCCESS: Image file created. \033[0m\n';
        else
            printf -- '\033[31m ERROR: Cannot create image. \033[0m\n';
            exit 9;
        fi
    fi

    # generate image from sandbox
    sudo singularity build $LOCAL_IMAGE $LOCAL_IMAGE_SANDBOX;
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: Image file created. \033[0m\n';
    else
        printf -- '\033[31m ERROR: Cannot create image. \033[0m\n';
        exit 8;
    fi

    # update the run script
    sudo rm -f run_chemicalchecker.sh;
    wget $REMOTE_CCREPO/run_chemicalchecker.sh;
fi
