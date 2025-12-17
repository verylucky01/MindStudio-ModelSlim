#!/usr/bin/env bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export ASCEND_RT_VISIBLE_DEVICES=0

declare -i ret_ok=0
declare -i ret_failed=1
run_ok=$ret_ok

source /home/ptq-test/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate llm_ptq_ra

rm -rf $PROJECT_PATH/output/ra_compression_llama/*
python run.py
if [ $? -eq 0 ]
then
    echo ra_compression_llama: Success
else
    echo ra_compression_llama: Failed
    run_ok=$ret_failed
fi

conda activate smoke_test_modelslim_0104

exit $run_ok