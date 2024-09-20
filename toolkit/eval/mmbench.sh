#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CKPT=$1
SPLIT=$2

output_dir=$SCRIPT_DIR/../../output/eval_results/$CKPT/mmbench

CUDA_VISIBLE_DEVICES=$3 python -m mgm.eval.model_vqa_mmbench \
    --model-path $SCRIPT_DIR/../../output/training_dirs/$CKPT \
    --question-file $SCRIPT_DIR/../training/data/eval_stage_1/mmbench/$SPLIT.tsv \
    --answers-file $output_dir/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode gemma

mkdir -p $output_dir/answers_upload/$SPLIT

python $SCRIPT_DIR/../training/scripts/convert_mmbench_for_submission.py \
    --annotation-file $SCRIPT_DIR/../training/data/eval_stage_1/mmbench/$SPLIT.tsv \
    --result-dir $output_dir/answers/$SPLIT \
    --upload-dir $output_dir/answers_upload/$SPLIT \
    --experiment $CKPT
