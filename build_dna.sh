#!/bin/bash

TMP=$(uuidgen).pdb
TMPD="/tmp/$(uuidgen)"
OCWD=$(pwd)

mkdir ${TMPD}
cd ${TMPD}

fiber -b -seq=${1} -single ${TMP}
sed -i '1,4d' ${TMP}

gmx pdb2gmx -f ${TMP} -ff amber99sb-ildn -water none -o ${1}.pdb

cd ${OCWD}
cp ${TMPD}/${1}.pdb .

rm -rf ${TMPD}
