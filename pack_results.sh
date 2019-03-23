#!/bin/bash

INPUT="/home/ubuntu/INFO259"
OUTPUT="/home/ubuntu/INFO259"

if [ ! -d ${INPUT}/all ]; then
    mkdir ${INPUT}/all
    mkdir ${INPUT}/all/pics
fi

cp ${INPUT}/pics/*.png ${OUTPUT}/all/pics/.
cp ${INPUT}/*.csv ${OUTPUT}/all/.
tar -zcf all.tar.gz all
