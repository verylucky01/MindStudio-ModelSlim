#!/usr/bin/env/bash

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

source /opt/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate modelslim_py310

rm -rf $PROJECT_PATH/output/ptq-tools/*
torchrun --nproc_per_node=1 run.py \
    $PROJECT_PATH/resource/multi_modal/opensora_project/sample-dsp.py \
    --num-frames 68 \
    --image-size 480 640 \
    --layernorm-kernel False \
    --flash-attn True \
    --sequence_parallel_size 1 \
    --prompt "A beautiful waterfall" \
    --save-dir $PROJECT_PATH/output/ptq-tools/samples/ \
    --sample-name a_beautiful_waterfall_dsp4

if [ $? -eq 0 ]
then
    echo quant_opensora_1_2: Success
else
    echo quant_opensora_1_2: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

# 清理output
rm -rf $PROJECT_PATH/output/ptq-tools/*

exit $run_ok