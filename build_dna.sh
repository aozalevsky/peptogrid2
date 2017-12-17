#!/bin/bash

TMP=$(uuidgen).pdb

fiber -b -seq=${1} -single ${TMP}
sed -i '1,4d' ${TMP}
gmx pdb2gmx -f ${TMP} -ff amber99sb-ildn -water none -o ${1}.pdb

rm ${TMP}
