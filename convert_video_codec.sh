#!/usr/bin/env bash

# :::::::::::::::::: Options ::::::::::::::::::
STUDY_ID_MIN=1
STUDY_ID_MAX=18
BITRATE=20
INPUT_SUFFIX="_stack"
OUTPUT_SUFFIX="_stack_sly"
# :::::::::::::::::::::::::::::::::::::::::::::

for STUDY_ID in $(seq $STUDY_ID_MIN $STUDY_ID_MAX); do
    STUDY_ID=$(printf "%02d" $STUDY_ID)
    INPUT_VIDEO="dataset/${STUDY_ID}/${STUDY_ID}${INPUT_SUFFIX}.mp4"
    OUTPUT_VIDEO="dataset/${STUDY_ID}/${STUDY_ID}${OUTPUT_SUFFIX}.mp4"
    ffmpeg -y -i $INPUT_VIDEO -vcodec libx264 -b:v ${BITRATE}M -f mp4 ${OUTPUT_VIDEO}
    printf "\n\033[1;92mStudy ${STUDY_ID} processed\n"
done