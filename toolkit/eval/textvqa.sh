#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

CKPT=$1
output_dir=$SCRIPT_DIR/../../output/eval_results/$CKPT/textvqa/

CUDA_VISIBLE_DEVICES=$2
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    python -m mgm.eval.model_vqa_loader \
        --model-path $SCRIPT_DIR/../../output/training_dirs/$CKPT \
        --question-file $SCRIPT_DIR/../training/data/eval_stage_1/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder $SCRIPT_DIR/../training/data/eval_stage_1/textvqa/train_images \
        --answers-file $output_dir/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode gemma &
done

wait

# Clear out the output file if it exists.
> "$output_dir/bm1.jsonl"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output_dir/${CHUNKS}_${IDX}.jsonl >> "$output_dir/bm1.jsonl"
done
