#!/bin/bash

PACKAGE_NAME=$1
FULL_VERSION=$2

MINOR_VERSION=${FULL_VERSION%.*}
MAJOR_VERSION=${FULL_VERSION%.*.*}

cd docs
make html
cd ..

#init_devpi.sh

devpi upload --with-docs
devpi test ${PACKAGE_NAME}==$FULL_VERSION
devpi push ${PACKAGE_NAME}==$FULL_VERSION root/dev
