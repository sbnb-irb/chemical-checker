#!/bin/sh

LOCAL_CCREPO=$HOME/local_checker

REMOTE_CCREPO=git@gitlab.sbnb.org/project-specific-repositories/chemical_checker.git

LOCAL_IMAGE=$LOCAL_CCREPO/cc.simg
LOCAL_IMAGE_SANDBOX=$LOCAL_CCREPO/sandbox
LOCAL_IMAGE_DEF=$LOCAL_CCREPO/cc.def

# display usage for current script
usage () {
    echo ""
    echo "Usage: $ bash setup_chemicalchecker.sh [-ieh]"
    echo ""
    echo "  -i      force update image"
    echo "  -e      edit config file"
    echo "  -d      path to source code for development mode"
    echo "  -h      print this help"
    echo ""
    exit 1
}

# compare versions
version_gt () {
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1";
}

# parse arguments
unset FORCE_UPDATE_IMAGE FORCE_CREATE_IMAGE
FORCE_UPDATE_IMAGE=false
FORCE_CREATE_IMAGE=true
FORCE_CHANGE_ENV=false
while getopts 'i:e:d:h' c
do
  case $c in
    i) FORCE_UPDATE_IMAGE=true ; FORCE_CREATE_IMAGE=false;;
    d) PATH_BRANCH=$OPTARG ; FORCE_CHANGE_ENV=true ; FORCE_CREATE_IMAGE=false ;;
    e) FORCE_CREATE_IMAGE=false;;
    h) usage ;; esac
done

echo $FORCE_UPDATE_IMAGE;
echo $FORCE_CREATE_IMAGE;
echo $FORCE_CHANGE_ENV;
echo $PATH_BRANCH

# check if needed binaries are available
_=$(command -v singularity);
if [ "$?" != "0" ]; then
    printf -- "\033[31m ERROR: You don't seem to have Singularity installed \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 127;
fi;

# check singularity version
SINGULARITY_MIN_VERSION=2.5.0
SINGULARITY_INSTALLED_VERSION="$(singularity --version)"
if version_gt $SINGULARITY_MIN_VERSION $SINGULARITY_INSTALLED_VERSION; then
    printf -- "\033[31m ERROR: Update Singularity, we require at least version $SINGULARITY_MIN_VERSION ($SINGULARITY_INSTALLED_VERSION detected) \033[0m\n";
    printf -- 'Follow the guide at: https://www.sylabs.io/guides/2.6/user-guide/installation.html\n';
    exit 127;
fi


_=$(command -v git);
if [ "$?" != "0" ]; then
    printf -- "\033[31m ERROR: You don't seem to have Git installed \033[0m\n";
    printf -- 'Follow the guide at: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git\n';
    exit 127;
fi;

# check if the chemical checker repository is available
if [ -d "$LOCAL_CCREPO" ]
then
    printf -- '\033[32m SUCCESS: Chemical Checker local directory available. (%s) \033[0m\n' $LOCAL_CCREPO;
    cd $LOCAL_CCREPO
else
    printf -- 'creating local directory. (%s) \n' $LOCAL_CCREPO;
    mkdir $LOCAL_CCREPO
    cd $LOCAL_CCREPO
    
fi

wget http://gitlab.sbnb.org/project-specific-repositories/chemical_checker/raw/master/container/singularity/cc-full.def -O cc.def
if [ $? -eq 0 ]; then
    printf -- '\033[32m SUCCESS: def file cloned. \033[0m\n';
else
    printf -- '\033[31m ERROR: could not clone def file. \033[0m\n';
    exit 1;
fi

if [ "$FORCE_CREATE_IMAGE" = true ]
then
    printf -- 'Removing old singularity image...\n';
    sudo rm $LOCAL_IMAGE;
    sudo rm -rf $LOCAL_IMAGE_SANDBOX;
    sudo mkdir $LOCAL_IMAGE_SANDBOX;
fi

# check if a local singularity image is available, otherwise copy
if [ -f "$LOCAL_IMAGE" ]
then
    sudo rm $LOCAL_IMAGE;
    printf -- '\033[32m SUCCESS: Cleaning old image available. (%s) \033[0m\n' $LOCAL_IMAGE;
    if [ "$FORCE_UPDATE_IMAGE" = true ]
    then
        sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX git --git-dir=/opt/chemical_checker/.git pull 
        printf -- '\033[32m SUCCESS: Pulling latest checker source code. \033[0m\n' ;
    fi
else
    printf -- 'Creating singularity sandbox image... \n';
    sudo singularity build --sandbox $LOCAL_IMAGE_SANDBOX $LOCAL_IMAGE_DEF
    if [ $? -eq 0 ]; then
        printf -- '\033[32m SUCCESS: image sandbox created. \033[0m\n';
    else
        printf -- '\033[31m ERROR: creating sandbox image. \033[0m\n';
        exit 2;
    fi
    
fi

if [ "$FORCE_CHANGE_ENV" = true ]
then
    text_replace_py="/export PYTHONPATH/c\\    export PYTHONPATH=\""$PATH_BRANCH"/package\":\$PYTHONPATH"
    text_replace_conf="/export CC_CONFIG/c\\    export CC_CONFIG=\""$PATH_BRANCH"/cc_config.json\""
    sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX sed -i "$text_replace_py" /environment
    sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX sed -i "$text_replace_conf" /environment
    printf -- '\033[32m SUCCESS: Changing image environment variables. \033[0m\n' ;
fi


printf -- 'Changing chemical checker config file.\n';
sudo singularity exec  --writable $LOCAL_IMAGE_SANDBOX vi /opt/chemical_checker/cc_config.json
printf -- 'Creating singularity checker image.\n';
sudo singularity build $LOCAL_IMAGE $LOCAL_IMAGE_SANDBOX
if [ $? -eq 0 ]; then
    printf -- '\033[32m SUCCESS: image created. \033[0m\n';
else
    printf -- '\033[31m ERROR: create image. \033[0m\n';
    exit 2;
fi
