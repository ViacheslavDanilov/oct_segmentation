#!/usr/bin/env bash

# :::::::::::::::::: Options ::::::::::::::::::
INPUT_DVC_PATH='dvc/data/raw.dvc'
INPUT_ZIP_PATH='data/raw.zip'
INPUT_DIR='data/raw'
INCLUDE_DIRS="[]"           # Usage example: INCLUDE_DIRS="['05','15','25']"
EXCLUDE_DIRS="[]"           # Usage example: EXCLUDE_DIRS="['07','13','20']"
OUTPUT_SIZE="[1000,1000]"
FPS=15
OUTPUT_DIR='data/sly_input'
# :::::::::::::::::::::::::::::::::::::::::::::

printf '\033[1;92mConversion pipeline:\n\033[0m'
printf '\033[0;94mInput path......: %s\n\033[0m' ${INPUT_PATH}
printf '\033[0;94mInput dir.......: %s\n\033[0m' ${INPUT_DIR}
printf '\033[0;94mIncluded dirs...: %s\n\033[0m' "${INCLUDE_DIRS}"
printf '\033[0;94mExcluded dirs...: %s\n\033[0m' "${EXCLUDE_DIRS}"
printf '\033[0;94mFPS.............: %d\n\033[0m' ${FPS}
printf '\033[0;94mOutput dir......: %s\n\033[0m' ${OUTPUT_DIR}

printf '\n\033[1;92mDownload source dataset from DVC...\n'
dvc pull ${INPUT_DVC_PATH}

# shellcheck disable=SC2236
if  [ ! -z  ${INPUT_ZIP_PATH}  ]
then
  printf '\n\033[1;92mUnzip source dataset...\n'
  unzip -n ${INPUT_ZIP_PATH} -d 'data'
  python src/data/reformate_data.py \
    meta.study_dir=${INPUT_DIR}
fi

printf '\n\033[1;92mConvert DICOMs to regular images...\n'
python src/data/convert_dicoms.py \
    convert.study_dir=${INPUT_DIR} \
    convert.include_dirs="${INCLUDE_DIRS}" \
    convert.exclude_dirs="${EXCLUDE_DIRS}" \
    convert.output_type='image' \
    convert.output_size="${OUTPUT_SIZE}" \
    convert.to_gray=false \
    convert.save_dir=${OUTPUT_DIR}

printf '\n\033[1;92mConvert DICOMs to grayscale images...\n'
python src/data/convert_dicoms.py \
    convert.study_dir=${INPUT_DIR} \
    convert.include_dirs="${INCLUDE_DIRS}" \
    convert.exclude_dirs="${EXCLUDE_DIRS}" \
    convert.output_type='image' \
    convert.output_size="${OUTPUT_SIZE}" \
    convert.to_gray=true \
    convert.save_dir=${OUTPUT_DIR}

printf '\n\033[1;92mStack images...\n'
python src/data/stack_images.py \
    stack.study_dir=${OUTPUT_DIR} \
    stack.include_dirs="${INCLUDE_DIRS}" \
    stack.exclude_dirs="${EXCLUDE_DIRS}" \
    stack.output_type='video' \
    stack.output_size="${OUTPUT_SIZE}" \
    stack.fps=${FPS} \
    stack.save_dir=${OUTPUT_DIR}

rm -f ${INPUT_ZIP_PATH}

printf '\n\033[1;92mComplete\n\033[0m'
